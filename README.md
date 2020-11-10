# Classfying Welding Defects Using Deep Learning

[Jupyter Notebooks](/Notebooks)

**PROJECT DESCRIPTION:**

Gas tungsten arc welding, also known as tungsten inert gas welding, is an arc welding process that uses a non-consumable tungsten electrode to produce the weld.

In TIG welding, there generally occurs these following cases:

1. Good weld
2. Burn through
3. Contamination
4. Lack of fusion
5. Misalignment
6. Lack of penetration

We will build a program that will detect if there’s any defect in the TIG welding process using Deep Learning. It will classify the defects as well if there are any.

So, our program will take an image, and it will give an output specifying any of the above 6 cases, whether the weld is good, it has burned through, etc.

**APPLICATION AREAS:**

This project can be further improved and integrate with the camera that can give industries that uses TIG welding or any other welding process to give them real-time feedback of the welding. This will be a very powerful solution for the industries which use automated welding.

This solution can save industries a whole sum of money.

**PLATFORM and OTHER REQUIREMENTS:**

We’ll be using Python-based Jupyter Notebook to design the model architecture. There exist many free to use cloud computing platforms, Google Colab is one of them. We’ll be using it to train the model.

There are available open datasets for TIG welding which can be used to train our model.

One such can be found here: [Link](https://www.kaggle.com/danielbacioiu/tig-aluminium-5083/)

## Tungsten Inert Gas (TIG) Welding

It is an arc welding process that uses a non-consumable tungsten electrode to produce the weld. The weld area and electrode are protected from oxidation or other atmospheric contamination by an inert shielding gas (argon or helium), and a filler metal is typically used.

TIG is used in joining high-value precision components. It is most commonly used to weld stainless steel, aluminum, magnesium, and copper alloys. This welding process allows greater control of welding to the welder, but at the same time, it is more challenging to master. Hence, there is room for errors in welding when it comes to TIG welding.

**Common TIG welding Defects**

- Poor gas coverage
- Dirty base/filler metal
- Improper arc length control

It is clear that the TIG welding process requires a lot of skill if done manually and hence welding of complex geometries is difficult to repeat, and the scrappage rates are high. Therefore it is crucial to automate the welding process. During the process, the monitoring and control of the weld pool are critical to the automation to ensure quality, consistency, and repeatability.

We propose to use a convolutional neural network to classify the quality of welding into six different categories. We plan to use the dataset available at Kaggle on TIG welding as our training dataset.

The weld can be classified into:

1. **Good weld**
2. **Burn Through:**When the weld metal completely penetrates the base metal.
3. **Contamination:**Contamination makes the welding porous, which in turn compromises the welding strength. There can be various reasons for contamination, such as contaminated metal, electrodes, or improper flow of shield gas.
4. **Lack of fusion:**It can be due to poor welding techniques. For example, if travel speed is too slow, the arc’s leading-edge will be in the puddle (it should be ahead of the puddle), resulting in a lack of fusion.
5. **Misalignment:**It can be due to poor component fit-up or relative motion between the components during the welding process.
6. **Lack of penetration:**Incomplete penetration is a weld bead that does not start at the root of the weld
groove. Generally, it is caused by too low a current level giving inadequate penetration.

# Sample Of each type of welding category

## Image processing for Model training

Given the image’s size, we process the image, remove the irrelevant aspects of the given image, and design the image suitable for our training. We have decided the figure below to be an initial start of image processing, but depending on the results we receive on our CNN model, we may tweak the image accordingly.

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image10.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image10.png)

## Modeling and Initial Model design

We have decided to use the CNN model to use for image classification. It is proven to have provided promising results with images. It is the best model to extract excellent features in an image, just like the subtle difference between various visual anomalies in TIG welding.

A CNN model watches a given image piece by piece picking up on features such as edges, corners, color variation patterns, creating a feature map, just like us, and then using those feature maps it decides the category the image falls into (in our case the six different classes of welding).

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image13.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image13.png)

This figure shows how a single feature map is created.

We will be using the model in the figure below (visualization of the feature map dimensions) in our initial modeling stages.

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image12.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image12.png)

## Description of the Material Used

The material used in welding tool piece was Aluminium 5083 Alloy with the following composition

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image3.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image3.png)

## Setup used for Capturing the Images:

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image4.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image4.png)

## The TIG welding Parameters

![Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image5.png](Classfying%20Welding%20Defects%20Using%20Deep%20Learning%208e332d945d1d4a5bba34415405825e40/image5.png)

## Description of the Dataset which contains the Images



| LABEL               | NUMBER OF SAMPLES |      |
| ------------------- | --------------------- | ---- |
|                     | Train                 | Test |
| Good weld           | 1102                  | 2189 |
| Burn through        | 892                   | 351  |
| Contamination       | 909                   | 2078 |
| Lack of fusion      | 1008                  | 1007 |
| Misalignment        | 988                   | 729  |
| Lack of penetration | 942                   | 234  |
| TOTAL               | 5841                  | 6588 |


### Dataset Containing Images:

### [https://drive.google.com/drive/folders/1-4-_7lkvVpkS-9dw6TC-daxTbngtcnjG?usp=sharing](https://drive.google.com/drive/folders/1-4-_7lkvVpkS-9dw6TC-daxTbngtcnjG?usp=sharing)

The following table contains information about the dataset in the Google Drive Link



| **Folders (Training Images)** | **Contains Image which has** |
| --------------------------------- | -------------------------------- |
| 170906-141809-Al 2mm-part1    | Good weld                        |
| 170906-120346-Al 2mm          | Good weld                        |
| 170906-114912-Al 2mm          | Good weld                        |
| 170913-152931-Al 2mm-part1    | Good weld                        |
| 170905-115602-Al 2mm          | Good weld                        |
| 170913-151508-Al 2mm-part1    | Good weld                        |
| 170905-114307-Al 2mm          | Good weld                        |
| 170913-155806-Al 2mm-part1    | Good weld                        |
| 170904-141730-Al 2mm-part1    | Good weld                        |
| 170913-143933-Al 2mm-part1    | Good weld                        |
| 170904-145718-Al 2mm-part1    | Good weld                        |
| 170913-142501-Al 2mm          | Good weld                        |
| 170906-144958-Al 2mm          | Burn through                     |
| 170906-113317-Al 2mm-part3    | Burn through                     |
| 170906-153326-Al 2mm-part2    | Contamination                    |
| 170904-151845-Al 2mm-part2    | Contamination                    |
| 170815-133921-Al 2mm          | Contamination                    |
| 170906-141809-Al 2mm-part2    | Contamination                    |
| 170904-141730-Al 2mm-part2    | Contamination                    |
| 170913-143933-Al 2mm-part2    | Contamination                    |
| 170904-141232-Al 2mm-part2    | Contamination                    |
| 170904-115959-Al 2mm          | Contamination                    |
| 170913-151508-Al 2mm-part2    | Contamination                    |
| 170913-152931-Al 2mm-part2    | Contamination                    |
| 170904-112347-Al 2mm          | Contamination                    |
| 170815-134756-Al 2mm          | Contamination                    |
| 170904-150144-Al 2mm-part1    | Contamination                    |
| 170906-151353-Al 2mm          | Lack of fusion                   |
| 170906-150801-Al 2mm          | Lack of fusion                   |
| 170906-150010-Al 2mm          | Lack of fusion                   |
| 170913-140725-Al 2mm          | Misalignment                     |
| 170905-110711-Al 2mm-part2    | Misalignment                     |
| 170904-155523-Al 2mm          | Misalignment                     |
| 170904-154202-Al 2mm-part2    | Misalignment                     |
| 170904-152301-Al 2mm-part1    | Misalignment                     |
| 170904-145718-Al 2mm-part2    | Misalignment                     |
| 170904-141730-Al 2mm-part3    | Misalignment                     |
| 170904-141232-Al 2mm-part3    | Misalignment                     |
| 170904-113012-Al 2mm-part2    | Misalignment                     |
| 170913-155806-Al 2mm-part2    | Misalignment                     |
| 170906-153326-Al 2mm-part1    | Lack of penetration          |
| 170904-151845-Al 2mm-part1    | Lack of penetration          |
| 170904-141232-Al 2mm-part1    | Lack of penetration          |
| 170904-115503-Al 2mm          | Lack of penetration          |
| 170904-113012-Al 2mm-part1    | Lack of penetration          |
| 170905-110711-Al 2mm-part1    | Lack of penetration          |

