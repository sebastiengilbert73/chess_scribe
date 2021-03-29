import cv2
import argparse
import logging
import os
import numpy as np
from tesserocr import PyTessBaseAPI
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('ImageFilepath', help="The filepath to the screenshot")
parser.add_argument('--outputDirectory', help="The directory where the output will be written. Default: '/tmp/chess_scribe/'", default='/tmp/chess_scribe/')
parser.add_argument('--rangeThreshold', help="The threshold on the grayscale range, for a row. Default: 100", type=int, default=100)
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main():
    logging.info("screenshot_to_csv.py main()")

    # If the output folder doesn't exist, create it. Cf. https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Open the image
    screenshot = cv2.imread(args.ImageFilepath, cv2.IMREAD_COLOR)
    screenshot_shapeHWC = screenshot.shape

    # Convert to grayscale
    grayscale_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find the text break rows
    text_line_delimiters = TextLineDelimiters(args.outputDirectory, grayscale_screenshot, args.rangeThreshold)
    logging.debug("text_line_delimiters = {}".format(text_line_delimiters))

    # Append text lines to form a single text line
    single_line_img = AppendTextLines(args.outputDirectory, screenshot, text_line_delimiters)

    #single_line_rgb = cv2.cvtColor(single_line_img, cv2.COLOR_BGR2RGB)
    with PyTessBaseAPI() as tesser_api:
        tesser_api.SetImage(Image.fromarray(single_line_img))
        logging.debug("tesser_api.GetUTF8Text() = {}".format(tesser_api.GetUTF8Text()))
    #text_str = pytesseract.image_to_string(Image.fromarray(single_line_rgb))



def TextLineDelimiters(output_directory, grayscale_screenshot, range_threshold):
    text_line_delimiters = [0]
    img_sizeHW = grayscale_screenshot.shape
    row_ranges = []
    for y in range(img_sizeHW[0]):
        min_value, max_value, _, _ = cv2.minMaxLoc(grayscale_screenshot[y, :])
        row_ranges.append(max_value - min_value)

    with open(os.path.join(output_directory, "TextLineDelimiters_rowRange.csv"), 'w+') as stats_file:
        stats_file.write("y,range\n")
        we_are_in_text = False
        for y in range(len(row_ranges)):
            grayscale_range = row_ranges[y]
            stats_file.write("{},{}\n".format(y, grayscale_range))
            if grayscale_range >= range_threshold:
                we_are_in_text = True
            else:
                if we_are_in_text:
                    text_line_delimiters.append(y)
                we_are_in_text = False
    return text_line_delimiters

def AppendTextLines(output_directory, screenshot, text_line_delimiters):
    deltas = [text_line_delimiters[i] - text_line_delimiters[i - 1] for i in range(1, len(text_line_delimiters))]
    text_line_height = max(deltas)
    deltas.append(text_line_height)
    logging.debug("text_line_height = {}".format(text_line_height))
    text_line_width = screenshot.shape[1] * len(text_line_delimiters)
    single_line_img = np.zeros((text_line_height, text_line_width, 3), dtype=np.uint8)
    for lineNdx in range(len(text_line_delimiters) - 1):
        #logging.debug("lineNdx  = {}; text_line_delimiters[lineNdx] = {}; deltas[lineNdx] = {}".format(lineNdx, text_line_delimiters[lineNdx], deltas[lineNdx]))
        single_line_img[0: deltas[lineNdx], lineNdx * screenshot.shape[1] : (lineNdx + 1) * screenshot.shape[1]] = \
        screenshot[text_line_delimiters[lineNdx]: text_line_delimiters[lineNdx] + deltas[lineNdx], :]

    single_line_filepath = os.path.join(output_directory, "AppendTextLines_singleLine.png")
    cv2.imwrite(single_line_filepath, single_line_img)
    return single_line_img

if __name__ == '__main__':
    main()