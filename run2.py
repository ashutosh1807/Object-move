from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
import cv2
import argparse
from grounded_sam import grounded_segmentation, get_masks


def shift_array(array, shift_x, shift_y):
        shifted_array = np.zeros_like(array)
        orig_y_range = slice(max(0, -shift_y), min(array.shape[0], array.shape[0] - shift_y))
        orig_x_range = slice(max(0, -shift_x), min(array.shape[1], array.shape[1] - shift_x))
        
        shifted_y_range = slice(max(0, shift_y), min(array.shape[0], array.shape[0] + shift_y))
        shifted_x_range = slice(max(0, shift_x), min(array.shape[1], array.shape[1] + shift_x))
        
        if array.ndim == 2:  
            shifted_array[shifted_y_range, shifted_x_range] = array[orig_y_range, orig_x_range]
        elif array.ndim == 3: 
            shifted_array[shifted_y_range, shifted_x_range, :] = array[orig_y_range, orig_x_range, :]
        
        return shifted_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--class', dest='class_name', required=True, help="Class name for the image")
    parser.add_argument('--output', required=True, help="Path to save the output image")
    parser.add_argument('--x', required=True, help="Object Move in x direction")
    parser.add_argument('--y', required=True, help="Object Move in y direction")
    args = parser.parse_args()
    

    image_path = args.image
    class_name = args.class_name
    output_path = args.output
    labels = [class_name]
    shift_x = int(args.x)
    shift_y = int(args.y)
    
    threshold = 0.3
    
    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-huge"
    simple_lama = SimpleLama()

    image_array, detections = grounded_segmentation(
        image=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=False,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    #Generate mask
    image = Image.open(image_path)
    mask = Image.fromarray(get_masks(image_array, detections)).convert('L')
    new_width = image.width 
    new_height = image.height 

    #dilate mask
    mask_array = np.array(mask)
    kernel = np.ones((5,5), np.uint8) 
    dilated_mask = cv2.dilate(mask_array, kernel, iterations=10)
    dilated_mask_image = Image.fromarray(dilated_mask)

    #remove object
    result = simple_lama(image, dilated_mask_image)
    result_resized = result.resize((new_width, new_height), Image.LANCZOS)

    #Extract object from image
    obj = image.copy()
    obj = np.array(obj)
    obj[mask_array==0] = 0

    #shift original mask and object
    shifted_mask = shift_array(np.array(mask), shift_x, shift_y)
    shifted_obj = shift_array(obj, shift_x, shift_y)

    #paste object on image
    result_resized = np.array(result_resized)
    result_resized[shifted_mask != 0] = shifted_obj[shifted_mask != 0]
    result_resized = cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR)

    #gerenate mask around the object for seamless object paste
    _, binary_mask = cv2.threshold(shifted_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10,10), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=10)
    dilated_mask[shifted_mask!=0] = 0

    #inpaint
    i = Image.fromarray(result_resized)
    m = Image.fromarray(dilated_mask)
    result = simple_lama(i, m)
    cv2.imwrite(output_path, np.array(result))

if __name__ == "__main__":
    main()