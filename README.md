# Programming Club Secretary Recruitment Task-2
## - by Pratul Koolwal
## - Roll No. - 230782

The Python Program is present inside the folder named "Task 2", with the name "task_2_Linear_Regression.py"

## Finding Dataset
In this task, dataset had to be found ourselves. but I could not find any reliable source to find data for this exact problem. So I used research statistics from ScienceDirect journal. there I found the Mean and standard Deviation of different parameters.
![Screenshot 2024-05-23 215912](https://github.com/Pratul9999/Pratul_Koolwal-Programming-Club-Secretary-Recruitment-Task-2/assets/152211763/6a5e2681-8675-4a11-9619-f570caefd9f3)

I use Python Libraries (namely pandas, numpy and scipy) to create my dataset using Mean, standard deviation, Upper and lower bounds of different parameters. I also used the correlation matrix from the article so that my dataset has more correlated values and not just random stuff.

The approach to create dataset uses Normal Distribution (actually truncated normal distribution to ensure lower and upper bounds) to ensure a more realistic dataset. So indirectly it uses Central limit theorem.

Link to the article - https://www.sciencedirect.com/science/article/pii/S1746809422003238?via%3Dihub#bi005

## My method
In this problem, I applied Linear Regression to predict Lung Tidal Volume using the dataset. Linear Regression is a basic, yet a powerful method in Machine Learning Models. In this case it is logical to expect near-linear relations between different parameters and Lung tidal volume, hence Linear regression appears to be a suitable choice.

At first, my model was generating way too random outcomes with very low r2 score. Then I realised this can be due to my data being very random due to my method of dataset generation. Then I decided to use Correlation matrix to ensure the dataset was a bit more accurate. Entries in Correlation matrix were taken from the above cited article itself. This significantly improved r2 score of my model.

I used Sci-kit learn library to implement this linear regression model.

## Instructions to run the program
The Python Program is present inside the folder named "Task 2", with the name "task_2_Linear_Regression.py". Sinces it creates its own dataset during program execution, it does not need any other companion file. So this file can be directly run on its own.

## Results
In trial runs, the model predicted Lung tidal volume wth a mean square error in the range 500-600 (occasionally reaching 700).
At the same time, The r2 score generally remained above 0.85.
![Screenshot 2024-05-23 233847](https://github.com/Pratul9999/Pratul_Koolwal-Programming-Club-Secretary-Recruitment-Task-2/assets/152211763/870e5d5e-e4ab-4308-8794-beb976971e7b)

![Screenshot 2024-05-23 233817](https://github.com/Pratul9999/Pratul_Koolwal-Programming-Club-Secretary-Recruitment-Task-2/assets/152211763/3f1acfe1-b148-431a-a26c-c58291499d64)

![Screenshot 2024-05-23 233751](https://github.com/Pratul9999/Pratul_Koolwal-Programming-Club-Secretary-Recruitment-Task-2/assets/152211763/3ee80a4f-fedf-4871-be75-fd38f964ec95)

Surely one can argue that this seems to be a constructed environment resulting in this performance of model, but I shall remind you that with dataset created using Mean, standard deviation, Upper and lower bounds, and a correlation matrix, it is always difficult to judge a model. So if we try to apply this model to a more realistic dataset, the model can probably perform better, and then it shall be worth it to apply some more optimisations to this approach.

Also, we can use other models like K-nearest Neighbours, or so, or we can go for more complex models of towards Neural Networks or deep learning to have even higher accuracy. But on the basis of the nature of this Problem, Linear Regression seems to do just fine.
