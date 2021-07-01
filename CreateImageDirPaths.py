import os

inputDir = "../IRLabeledDataset/ir_train_images_4" # directory containing input images
segmentationImageDir = "../IRLabeledDataset/ir_train_masks_4" # directory containing output segmentation images


def createImageDirPaths():

    if (not inputDir):
        print("The input dir is empty")
    if(not segmentationImageDir):
        print("The segmentation dir is empty")

    input_img_paths = sorted (
        [
            os.path.join(inputDir, fName)
            for fName in os.listdir(inputDir)
                if fName.endswith(".png")
        ]
    )

    target_img_paths = sorted (
        [
            os.path.join(segmentationImageDir, fName)
            for fName in os.listdir(segmentationImageDir)
                if fName.endswith(".png") and not fName.startswith(".")
        ]
    )

    print("Number of samples = {0}".format(len(input_img_paths)))

    for input_path, segmentation_mask_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", segmentation_mask_path)

    return input_img_paths, target_img_paths