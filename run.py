import cv2
import argparse
from grounded_sam import grounded_segmentation, plot_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--class', dest='class_name', required=True, help="Class name for the image")
    parser.add_argument('--output', required=True, help="Path to save the output image")

    args = parser.parse_args()

    image_path = args.image
    class_name = args.class_name
    output_path = args.output
    labels = [class_name]
    threshold = 0.3

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-huge"

    image_array, detections = grounded_segmentation(
        image=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=False,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    plotted_image = plot_detections(image_array, detections)
    plotted_image = cv2.cvtColor(plotted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, plotted_image)


if __name__ == "__main__":
    main()