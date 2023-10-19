# Telegram ML competition 2023

**The task in this competition is to create a library that detects a programming or markup language of a code snippet.**

____
My solution is inspired by [this article](https://arxiv.org/pdf/1509.01626.pdf) about char-level CNN for programming language classification

Steps:
* First of all I generated dataset with ~190k samples of labeled code snippets

The histplot below shows the amount of code snippets for each language (numbers on the X axis mathes with description in `common.py` file)
![FDCSL](https://github.com/mxm0312/libtglang/assets/21274627/1313e9ce-f347-4044-b3dd-85779fa5b184)

* Then I created a model and trained it on ~150k examples for 10 epochs
* After the model has trained, I got 79% accuracy on the validation dataset (~37k samples)
![train](https://github.com/mxm0312/libtglang/assets/21274627/edeb5d0e-6e60-42a8-8ccf-7f90a48a6298)
* Finally, I created telegram bot to test my model in real life

<table>
  <tr>
    <td valign="top"><img width="257" alt="photo_2022-09-26 00 12-2" src="https://github.com/mxm0312/libtglang/assets/21274627/a6c9d572-fa7d-4a73-bf2a-2f59615ba70c"></td>
     <td valign="top"><img width="258" alt="photo_2022-09-26 00 12-4" src="https://github.com/mxm0312/libtglang/assets/21274627/34dd6401-67e7-48c9-b714-50f805ac8d52"></td>
    <td valign="top"><img width="258" alt="photo_2022-09-26 00 12-4" src="https://github.com/mxm0312/libtglang/assets/21274627/5e26e169-755a-4e36-b396-79759f827ee8"></td>
  </tr>
 </table>
