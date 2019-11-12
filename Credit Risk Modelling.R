# Title : "Credit Risk Modelling - German Bank"
# Author: "Subramanian Kannappan"
# Date  :  9th Nov 2019


# NOTE: 
#    (1) This project is regarding the loan provided to the customers of German credit bank. Bank needs to
#        develop Credit Risk Modeling which will identify whether based on list of criteria whether the 
#        customerto will repay the loan or not. This will help the bank to boost the profit margin by reducing the 
#        its NPA ( Non performing asset ) and judicially use its cash to provide loan for those customers 
#        who will replay the loan promptly.
#
#    (2) Code comments has been kept to minimal only for the places where explanation is required. This has 
#        been done to avoid blotting of the number of lines in the script file  
#
#    (3) More detail on the code and the analysis of the output of method calls has been given in detail in
#        the PDF file generated from the rmd script. Please refer it for additional information 
#
#    (4) As the site needs login credential to access the dataset file, in the program the data set is not directly  
#        read from the site for processing. It is priorly downloaded to the local folder and then accessed in 
#        the program from the local folder for processing
#
#        If you wish you execute the code please do the following steps
#            (a) download the file from https://www.kaggle.com/kabure/german-credit-data-with-risk. 
#            (b) In the code replace the line "credit_data<- read_csv("C:\\Subbu\\german_credit_data.csv")"  
#                with the location where you are storing the file
#    
#    (5) After extensive analysis below 8 models have been short listed for building the Credit risk model   
#            1) Logistic Regression   
#            2) Decision Tree   
#            3) Random Forest   
#            4) Gradient Boosting   
#            5) Knn   
#            6) Linear Discriminant Analysis (LDA)   
#            7) Quadriatic Discriminant Analysis (QDA)   
#            8) Support Vector Machines (SVM)
#
#    (6) For the above algoritms shortlisted we will build the model twice   
#          (a) First time using the Base methods from the base library like Random Forest, 
#              gbm (gradient Boosting Model) etc. Here we split the data set only once into training and 
#              testing ( single fold approach)
#          (b) Seond time using the "train" function from the Caret library which performs Cross validation 
#              The train function by itself split the data set into training and testing multiple times 
#              (i.e multi fold approach)
#        Though cross validation methods are supposed to provide better results over the base methods still we will
#        compare them to validate the effectiveness of the train method from the caret function

# Installing the required packages
      if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
      if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
      if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
      if(!require(tree)) install.packages("tree", repos = "http://cran.us.r-project.org")
      if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
      if(!require(gmodels)) install.packages("gmodels", repos = "http://cran.us.r-project.org")
      if(!require(brnn)) install.packages("brnn", repos = "http://cran.us.r-project.org")
      if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
      if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
      if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
      if(!require(rocc)) install.packages("rocc", repos = "http://cran.us.r-project.org")
      if(!require(rminer)) install.packages("rminer", repos = "http://cran.us.r-project.org")


# Loading the required libraries
      library(tidyverse) 
      library(caret) 
      library(readr) 
      library(data.table)
      library(randomForest)
      library(rpart)
      library(rpart.plot)
      library(gmodels)
      library(brnn)
      library(xgboost)
      library(MASS)
      library(dplyr)
      library(corrplot)
      library(kernlab)
      library(rocc)
      library(ggplot2)
      library(rminer)
      library(e1071)
      library(reshape2)
      library(gbm)
      source("http://pcwww.liv.ac.uk/~william/R/crosstab.r")

# Data Loading
      credit_data<- read_csv("C:\\Subbu\\german_credit_data.csv")

# Checking the basic structure & content
      str(credit_data)

#Checking the sample content 
      credit_data[1:10, ]

# Removing the unwanted column X1 as it seems to be just the row number and not of any significance 
# during model building
      credit_data<- credit_data %>% dplyr::select(-X1)

      
      
# Analyze list of columns with NA and remove them if there is significant impact
# ------------------------------------------------------------------------------
      
# Identifying the total list of columns which has NA values
      names(which(sapply(credit_data, anyNA)))

# Total number of fields which has NA Values
      sum(is.na(credit_data))

# count of NAs in each column
      sapply(credit_data, function(count) sum(is.na(count))) %>% 
        knitr::kable(col.names = c("\'NA\' Value counts"))

# Total number of rows which has NA Values
      sum(!complete.cases(credit_data))

# Removing the records with NA values from the dataset
      na_credit_data<- na.omit(credit_data)

# Crosschecking from the count whether all NA values has been removed
      nrow(na_credit_data)

# Renaming the columns with space between the names as space in their names as some of the builtin 
# functions throws error
      setnames(na_credit_data, old=c("Credit amount"), new=c("creditamount"))
      setnames(na_credit_data, old=c("Saving accounts"), new=c("savingsaccounts"))
      setnames(na_credit_data, old=c("Checking account"), new=c("checkingaccount"))    

# Number of unique values in each column after removing NA values. 
      apply(na_credit_data, 2, function(x) length(unique(x))) %>% 
        knitr::kable(col.names = c("Unique Value counts"))


      
# List of different values for the "discrete" value features
# -----------------------------------------------------------
      
      unique(na_credit_data$savingsaccounts) %>% 
        knitr::kable(col.names = c(" Savings Accounts Categories"))
      
      unique(na_credit_data$checkingaccount) %>% 
        knitr::kable(col.names = c("Checkig Accounts Categories"))
      
      unique(na_credit_data$Housing) %>% 
        knitr::kable(col.names = c("Housings Categories"))
      
      unique(na_credit_data$Purpose) %>% 
        knitr::kable(col.names = c("List  of Purposes"))


# Unique values of the numberic field "Job"
      sort(unique(credit_data$Job)) %>% 
        knitr::kable(col.names = c("Job Types"))

      
      
# Check for Correlation and Prevalence in the dataset
# ---------------------------------------------------      
      
# Checking for Correlation between numeric variables and remove if any is observed
      credit_data %>% 
        dplyr::select(Age, Job, creditamount, Duration) %>% cor() %>% knitr::kable()

# Checking for Prevalence issue in "Risk" feature in the original data set
      ct_risk <- CrossTable(credit_data$Risk)
      ct_risk$t %>% knitr::kable()

# Checking for "Prevalence" issue in "Risk" data post remvoing the NA values
      ct_risk_na <- CrossTable(na_credit_data$Risk) 
      ct_risk_na$t %>% knitr::kable()


# Checking for prevalence in Job data
      ggplot(na_credit_data, aes(Job) ) + 
        geom_bar(aes(fill = as.factor(na_credit_data$Job))) + 
        scale_fill_discrete(name="Job type",
                            labels=c( "Unskilled and Non-Resident","Unskilled and Resident", 
                                      "Skilled", "Highly Skilled")) + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
        labs(x= "Level of Job",y= "Frequency" , title = "Distribution of Job")

# Distribution of job data in terms of percentage
      CrossTable(na_credit_data$Job, format = 'SPSS', digits = 2)

# Plotting the distribution of Job against Credit Risk
      ggplot(na_credit_data, aes(Job) ) + 
        geom_bar(aes(fill = as.factor(na_credit_data$Job))) + 
        scale_fill_discrete(name="Job type",
                            labels=c( "Unskilled and Non-Resident","Unskilled and Resident", 
                                      "Skilled", "Highly Skilled")) + 
        theme(axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
        labs(x= "Level of Job",y= "Frequency" , title = "Distribution of Job v/s Credit Risk") +
        geom_text(stat='count', aes(label=..count..), vjust = -1) + 
        facet_grid( . ~ Risk)

# Distribution of Job against Credit Risk in terms of percentage
      CrossTable(na_credit_data$Job, na_credit_data$Risk, 
                 prop.c = FALSE , prop.t = FALSE , prop.chisq = FALSE , 
                 format = 'SPSS', digits = 2)

# Checking the Distribution of Male / Female ratio
      CrossTable(na_credit_data$Sex, format = 'SPSS', digits = 2)

# Counts on no.of occurences based on the 3 predictors Sex, Job & Risk category Split on the 
      crosstab(na_credit_data, row.vars = c("Sex", "Job"), 
               col.vars = "Risk", type = "frequency")

# Distribution of data based on the 3 features in terms of percentage
      crosstab(na_credit_data, row.vars = c("Sex", "Job"), col.vars = "Risk", type = "t") 

# Plotting the distribution of Job v/s Credit Risk v/s Sex
      ggplot(na_credit_data, aes(Job) ) + 
        geom_bar(aes(fill = as.factor(na_credit_data$Job))) + 
        scale_fill_discrete(name="Job Type",
                            labels=c( "Unskilled and Non-Resident","Unskilled and Resident", 
                                      "Skilled", "Highly Skilled")) + 
        theme(axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
        labs(x= "Job Type",y= "Frequency" , title = "Distribution of Job v/s Credit Risk v/s Sex") + 
        geom_text(stat='count', aes(label=..count..), vjust = -1) + 
        facet_grid( Sex ~ Risk)
      

# Distribution of Housing v/s Credit risk data   
      CrossTable(na_credit_data$Housing, na_credit_data$Risk, 
                 prop.c = FALSE , prop.t = FALSE , prop.chisq = FALSE , 
                 format = 'SPSS', digits = 2)

# Graph plot for Housing distribution v/s Credit risk in the data set
      ggplot(na_credit_data, aes(Housing) ) + 
        geom_bar(aes(fill = as.factor(na_credit_data$Housing))) + 
        scale_fill_discrete(name="Housing Type", labels=c( "Free","Own", "Rent")) + 
        theme(axis.text.x=element_blank(),axis.ticks.x=element_blank()) +
        labs(x= "Housing Type",y= "Frequency" , title = "Distribution of Housing v/s Credit Risk") +
        geom_text(stat='count', aes(label=..count..), vjust = -1) + 
        facet_grid( . ~ Risk)
      

# Converting the credit amount column to range of 2500 dollars each . This grouping is done as there are 
# large number of unique values becuase of it is difficult to check for Prevalence issue. By grouping 
# them in the range of 2500 we can check for prevalence in the range
      
      credit_range  <- 
        function(n) {  
          if(n %% 2500 > 0)  
            paste("Credit(",((floor(n/2500))*2500)+1,"-",(floor(n/2500)+1)*2500, ")" ,sep="")  
          else  
            paste("Credit(",((floor(n/2500)-1)*2500)+1,"-",(floor(n/2500))*2500 ,")", sep="") 
        }
      
      na_credit_data<- na_credit_data %>% 
        mutate(CreditRange = sapply(creditamount, credit_range))

# Changing the factor values sequence to get them in ascending order for our Cross table output
      na_credit_data$CreditRange <- 
        factor(na_credit_data$CreditRange, 
               levels = names(sort(table(na_credit_data$CreditRange), decreasing = TRUE) ))

# Distribution of Credit amount data
      CrossTable(na_credit_data$CreditRange, na_credit_data$Risk, 
                 prop.c = FALSE , prop.t = FALSE , prop.chisq = FALSE , 
                 format = 'SPSS', digits = 2) 
      

# Checking for prevalence in Age data
# -----------------------------------

# Similar to credit amount field we have issue in "age" column also . Hence for the sake of checking
# the prevalence issue we will group the records in the range of 10 years gap
      age_range<- function(n) {
        if(n %% 10 > 0)  
          paste("Age(",((floor(n/10))*10)+1,"-",(floor(n/10)+1)*10, ")" ,sep="")  
        else  
          paste("Age(",((floor(n/10)-1)*10)+1,"-",(floor(n/10))*10 ,")", sep="") 
      }
      
      na_credit_data<- na_credit_data %>% 
        mutate(AgeRange = sapply(Age, age_range))
      
      CrossTable(na_credit_data$AgeRange, na_credit_data$Risk, 
                 prop.c = FALSE , prop.t = FALSE , prop.chisq = FALSE , format = 'SPSS', digits = 2)


# Plotting a graph with the above age and credit amount 
      Age <-  na_credit_data$Age
      CreditAmount <- na_credit_data$creditamount 
      Risk <- na_credit_data$Risk
      
      qplot( Age, CreditAmount, colour = Risk ) +  
        scale_x_continuous(breaks = c(seq(11,80,10))) +
        scale_y_continuous(breaks = c(seq(0,20000,2500)))


# Checking for prevalence in Duration data
#-----------------------------------------      

# Like credit amount and Age column we will group the Durations column also to range of 12 months each and 
# check for prevalence issue in that Duration range      
      duration_range <- 
        function(n) { 
          if(n %% 12 > 0)  
            paste("Months(",((floor(n/12))*12)+1,"-",(floor(n/12)+1)*12, ")" ,sep="")  
          else  
            paste("Months(",((floor(n/12)-1)*12)+1,"-",(floor(n/12))*12 ,")", sep="") 
        }
      
      na_credit_data <- 
        na_credit_data %>% 
        mutate(MonthsRange = sapply(Duration, duration_range))
      
      CrossTable(na_credit_data$MonthsRange, na_credit_data$Risk, 
                 prop.c = FALSE , prop.t = FALSE , prop.chisq = FALSE , format = 'SPSS', digits = 2) 
      
      
# Plotting a graph with above Duration and credit amount
      Duration <- na_credit_data$Duration
      CreditAmount <- na_credit_data$creditamount 
      Risk <- na_credit_data$Risk
      
      qplot(Duration, CreditAmount, colour = Risk ) +  
        scale_x_continuous(breaks = c(seq(0,90,12))) +
        scale_y_continuous(breaks = c(seq(0,20000,2500)))  


# Removing those credits which has loan duration of more than 4 years ( 49-60 and 61-72 Months) as in the
# current data set all of them are bad credits and using them for modelling will not yield any useful result
      na_credit_data <- na_credit_data %>% filter(Age < 71)
      na_credit_data <- na_credit_data %>% filter(creditamount < 12501)
      na_credit_data <- na_credit_data %>% filter(Duration < 49)

# Generating QPlot - post removing the values
      Duration <- na_credit_data$Duration
      CreditAmount <- na_credit_data$creditamount 
      Risk <- na_credit_data$Risk
      
      qplot(Duration, CreditAmount, colour = Risk ) +  
        scale_x_continuous(breaks = c(seq(0,90,12))) +
        scale_y_continuous(breaks = c(seq(0,20000,2500)))      

#Remove the redunt columns of Age, Duration and Credit amount
      na_credit_data <- na_credit_data %>% 
        dplyr::select ( -contains("Range") ) 

# Changing the character columns to factors 
      na_credit_data$Risk <- as.factor(na_credit_data$Risk)
      na_credit_data$Sex <- as.factor(na_credit_data$Sex)
      na_credit_data$Housing <- as.factor(na_credit_data$Housing)
      na_credit_data$Purpose <- as.factor(na_credit_data$Purpose)
      na_credit_data$savingsaccounts <- 
        as.factor(na_credit_data$savingsaccounts)
      na_credit_data$checkingaccount <- 
        as.factor(na_credit_data$checkingaccount)


# Model Building
# ----------------      
# First we shall build Models using the base method available from respective libraries. After that we 
# shall build with "train" methods from Caret package
      
# Creating data partition - Segregating the data set into training and testing purpose
      set.seed(1250) 
      train_index<- createDataPartition(y = na_credit_data$Risk, 
                                        times = 1, p = 0.9, list = FALSE) 
      train_data<- na_credit_data [train_index,] 
      test_data<- na_credit_data [-train_index,]


      
# Model Building using base methods from respective libraries
# -------------------------------------------------------------
        
# Dataframe to store the results of different models for comparisions
      df_gen_models <- data.frame(matrix(vector(),ncol=4))
      colnames(df_gen_models) <- c("Model", "Accuracy", "Sensitivity", "Specificity" ) 


# Logistic Regression Model
# --------------------------
        
# Building the Logistic Regression Model
      glm_gen_model <- glm(Risk~ ., data = train_data, family = 'binomial')
      glm_gen_predict<- predict(glm_gen_model, test_data, type = 'response')

# Data frame to store the results of different cutoff value of Logistic regression
      df_glm_gen_model <- data.frame(matrix(vector(),ncol=5))
      colnames(df_glm_gen_model) <- c("Model","Cutoff", "Accuracy", "Sensitivity", "Specificity" ) 

# Function to evaluate and store the results of Logistic regression model  for different cutoff values
      glmfun <- function(a) { 
        glm_gen_predict_val<- factor(ifelse(glm_gen_predict>= a, "bad", "good"))
        glm_gen_cf <- confusionMatrix( glm_gen_predict_val, test_data$Risk)
        df_glm_gen_model <<- 
          df_glm_gen_model %>% 
          add_row(Model = "Logistic Regression",
                  Cutoff = a, 
                  Accuracy = glm_gen_cf$overall['Accuracy'] , 
                  Sensitivity = glm_gen_cf$byClass['Sensitivity'],
                  Specificity =  glm_gen_cf$byClass['Specificity'])
      }
      
      seqval<- seq(0.20,.90, 0.05)
      glm_output <- sapply(seqval,glmfun)
      
      df_glm_gen_model %>% dplyr::select(-Model) %>% knitr::kable()

# Transforming the values to different data frame format for graphical representation
      df_glm_gen_model_tmp <- df_glm_gen_model
      df_glm_gen_model_tmp <- gather(df_glm_gen_model_tmp, Type, Val, Accuracy:Specificity)
      df_glm_gen_model_tmp <- df_glm_gen_model_tmp [order(df_glm_gen_model_tmp$Cutoff), ]

# Graphical plot for the Logistic regression model for different cutoff values
      ggplot(df_glm_gen_model_tmp, aes(Cutoff, Val, color = Type)) + 
        geom_line() +
        geom_point(size = 3) + 
        scale_y_continuous(labels = scales::percent) + 
        scale_x_continuous(breaks = seq(0.1, 0.9, by = 0.05)) + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1), 
              panel.grid.minor.x = element_blank()) + 
              labs(y = "Accuracy Rate", 
                   title = "Logistic Regression Graph (general Model) for different cutt offs")

# Listing out the features that made significant impact on the model building
      glm_gen_model_vi <- as.data.frame(varImp(glm_gen_model))
      glm_gen_model_vi <- data.frame( feature  = rownames(glm_gen_model_vi), 
                                      impact = glm_gen_model_vi$Overall)
      glm_gen_model_vi[order(glm_gen_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable() 

# Creating the data frame to store the results of different cutoff value for the finetuned version of  
# Logistic regression
      df_glm_gen_model_ft <- data.frame(matrix(vector(),ncol=5))
      colnames(df_glm_gen_model_ft) <- c("Model", "Cutoff", "Accuracy", "Sensitivity", "Specificity" ) 
      seqval<- seq(0.20,.90, 0.05)


# Building the finetuned Logistic Regression Model
      glm_gen_model_ft <- glm(Risk~ Duration + checkingaccount + savingsaccounts + Sex , 
                              data = train_data, family = 'binomial')
      glm_gen_predict_ft <- predict(glm_gen_model_ft, test_data, type = 'response')


# Function to evaluate and store the results of finetuned version of the Logistic regression model for 
# different cutoff values
      glmfun_ft <- 
        function(a) { 
          glm_gen_predict_val_ft <- factor(ifelse(glm_gen_predict_ft >= a, "bad", "good"))
          glm_gen_cf_ft<- confusionMatrix(glm_gen_predict_val_ft, test_data$Risk)
          df_glm_gen_model_ft <<- 
            df_glm_gen_model_ft %>% 
            add_row(Model = "GLM_FineTuned",
                    Cutoff = a, 
                    Accuracy = glm_gen_cf_ft$overall['Accuracy'] , 
                    Sensitivity = glm_gen_cf_ft$byClass['Sensitivity'],
                    Specificity =  glm_gen_cf_ft$byClass['Specificity'])
        }
      
      glm_ft_output <- sapply(seqval, glmfun_ft)
      df_glm_gen_model_ft %>% dplyr::select(-Model) %>%knitr::kable()


# Adding the Short listed values of the Logistic regression model to the comparision data frame
      df_gen_models <-  df_glm_gen_model %>% filter(Cutoff== 0.50) %>% 
        dplyr::select(-Cutoff)  %>% rbind(df_gen_models, .)


# Decision Tree Model
# ---------------------
        
# Building the Decision Tree Model
      dt_gen_model  <- rpart(Risk~., data=train_data)
      dt_gen_predict <- predict(dt_gen_model, test_data)
      dt_gen_predict <- factor(ifelse(dt_gen_predict[,"bad"] >= .50, "bad", "good"))
      dt_gen_cf<- confusionMatrix( dt_gen_predict, test_data$Risk)
      dt_gen_cf

# Listing the TOP 5 variables that made significant impact in  Decision tree model
      dt_gen_model_vi <- as.data.frame(varImp(dt_gen_model))
      dt_gen_model_vi <- data.frame(feature   = rownames(dt_gen_model_vi),
                                    impact = dt_gen_model_vi$Overall)
      dt_gen_model_vi[order(dt_gen_model_vi$impact,decreasing = T),][1:5,] %>%knitr::kable()


# Finetuning the Decision Tree model with the high impacting features
      dt_gen_model_ft  <- 
        rpart(Risk~ creditamount + Duration + checkingaccount + Purpose + savingsaccounts, 
              data=train_data)
      dt_gen_predict_ft <- predict(dt_gen_model_ft, test_data)
      dt_gen_predict_ft <- factor(ifelse(dt_gen_predict_ft[,"bad"] >= .50, "bad", "good"))
      dt_gen_cf_ft<- confusionMatrix( dt_gen_predict_ft, test_data$Risk)
      dt_gen_cf_ft


# Comparing the outcomes of both the vanila nad the finetune version of Decisition tree model we see that
# while th sensitivity is same in both the models the specificity is higher for the vanila version. So 
# we will short list the vanila version of Decision Tree model  for comparing with the other models
      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "CART",
                Accuracy = dt_gen_cf$overall['Accuracy'] ,
                Sensitivity = dt_gen_cf$byClass['Sensitivity'],
                Specificity =  dt_gen_cf$byClass['Specificity'] )


# Random Forest Model
# -------------------
# NOTE : This model took around 10 min ( 6 GB + i3 processor) for the data set of 502 records ( post 
#        removing the outliers). If this model is selected in the final comparision then this 
#        performance factor has to be taken into consideration before deploying it for production purpose
      
# Bulding the Random Forest Model
      rf_gen_model  <- randomForest(Risk~., data=train_data)
      rf_gen_predict <- predict(rf_gen_model, test_data)
      rf_gen_cf<- confusionMatrix( rf_gen_predict, test_data$Risk)
      rf_gen_cf 

# Finding out the list of variables that made significant impact in the outcome of the Random Forest model
      rf_gen_model_vi <- as.data.frame(varImp(rf_gen_model))
      rf_gen_model_vi <- data.frame(feature   = rownames(rf_gen_model_vi),
                                    impact = rf_gen_model_vi$Overall)
      rf_gen_model_vi[order(rf_gen_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable()

# Finetuning the Random Forest model with the high impacting features
      rf_gen_model_ft  <- rpart(Risk~ creditamount + Age + Duration +  Purpose + Job, data=train_data)
      rf_gen_predict_ft <- predict(rf_gen_model_ft, test_data)
      rf_gen_predict_ft <- factor(ifelse(rf_gen_predict_ft[,"bad"] >= .50, "bad", "good"))
      rf_gen_cf_ft<- confusionMatrix( rf_gen_predict_ft, test_data$Risk)
      rf_gen_cf_ft

# Comparing the outcomes of both the vanila and the finetune version of Random Forest model we see that 
# sensitivity is higher for the vanila version. So for Random Forest method also  we will short list the 
# vanila version for comparing with the other models

      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "Random Forest",
                Accuracy = rf_gen_cf$overall['Accuracy'] ,
                Sensitivity = rf_gen_cf$byClass['Sensitivity'],
                Specificity =  rf_gen_cf$byClass['Specificity'])

            
# Building the Gradient Bossting Model
# -------------------------------------      
      
# Building the Gradient Bossting Model
      gbm_gen_model <- gbm(Risk ~., data = train_data, distribution = "multinomial")
      gbm_gen_predict <- predict(gbm_gen_model, test_data ,n.trees = 100,type = "response")
      gbm_gen_labels = colnames(gbm_gen_predict)[apply(gbm_gen_predict, 1, which.max)]
      gbm_gen_result = data.frame(test_data$Risk, gbm_gen_labels)
      gbm_gen_cf = confusionMatrix(test_data$Risk, as.factor(gbm_gen_labels))
      gbm_gen_cf

# Finding out the high impacting features of the Gradient Boosting model
      summary(gbm_gen_model)

# Finetuning the Gradient boosting model by the TOP 5 high impacting features
      gbm_gen_model_ft <- gbm(Risk ~ creditamount + Duration   + Age + Purpose + savingsaccounts, 
                              data = train_data, distribution = "multinomial")
      gbm_gen_predict_ft <- predict(gbm_gen_model_ft, test_data, n.trees = 100,type = "response")
      gbm_gen_labels_ft = colnames(gbm_gen_predict)[apply(gbm_gen_predict_ft, 1, which.max)]
      gbm_gen_result_ft = data.frame(test_data$Risk, gbm_gen_labels_ft)
      gbm_gen_cf_ft = confusionMatrix(test_data$Risk, as.factor(gbm_gen_labels_ft))
      gbm_gen_cf_ft

# In Gradient boosting model also we see that Sensitivity of the vanila version is higher than the 
# finetuned version. Hence vanila version will be short listed for this model for comparing with the 
# other models
      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "Gradient Boosting",
                Accuracy = gbm_gen_cf$overall['Accuracy'] ,
                Sensitivity = gbm_gen_cf$byClass['Sensitivity'],
                Specificity =  gbm_gen_cf$byClass['Specificity'])


# KNN Model
# -----------
        
# Dataframe to store outcomes of KNN model for various values of K
      df_knn_gen_models <- data.frame(matrix(vector(),ncol=5))
      colnames(df_knn_gen_models) <- c("Model", "K", "Accuracy", "Sensitivity", "Specificity" ) 

# Function to evaluate and store the results of KNN model for different "K" values
      knnfun <- function(a) { 
        knn_gen_model  <- knn3(Risk~., data=train_data, k = a)
        knn_gen_predict <- predict(knn_gen_model, test_data)
        knn_predict_val <- factor(ifelse(knn_gen_predict[,"bad"] >= .50, "bad", "good"))
        knn_gen_cf<- confusionMatrix( knn_predict_val, test_data$Risk)
        df_knn_gen_models <<- 
          df_knn_gen_models  %>% 
          add_row(Model = "KNN",
                  K = a,
                  Accuracy = knn_gen_cf$overall['Accuracy'] ,
                  Sensitivity = knn_gen_cf$byClass['Sensitivity'],
                  Specificity =  knn_gen_cf$byClass['Specificity'])
      }
      
      knn_output <- sapply(seq(2,9,1), knnfun)
      
      df_knn_gen_models[order(df_knn_gen_models$Sensitivity,
                              df_knn_gen_models$Specificity,decreasing = T),] %>% 
        dplyr::select(-Model)  %>% knitr::kable()
      

# For K=2 4 & 6 though the Sensitivity is high we see that Specifisity is very less which means that the 
# bank will be losing considerable potential customer and there by business. On the other hand for K = 5 
# both Sensitivity and Specificity are significantly good. So from the KNN model we will choose the 
# outcome of K = 5 for comparing with other models

      df_gen_models <-  df_knn_gen_models %>% filter(K==5) %>% 
        dplyr::select(-K) %>% rbind(df_gen_models, .)



# Linear Discriminant Analysis (LDA) Model
# ----------------------------------------      
      lda_gen_model  <- lda(Risk~., data=train_data)
      lda_gen_predict <- predict(lda_gen_model, test_data)$class
      lda_gen_cf<- confusionMatrix( lda_gen_predict, test_data$Risk)
      lda_gen_cf
      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "LDA",
                Accuracy = lda_gen_cf$overall['Accuracy'] ,
                Sensitivity = lda_gen_cf$byClass['Sensitivity'],
                Specificity =  lda_gen_cf$byClass['Specificity'] )

# Quadriatic Discriminant Analysis (QDA) Model
# --------------------------------------------
      qda_gen_model  <- qda(Risk~., data=train_data)
      qda_gen_predict <- predict(qda_gen_model, test_data)$class
      qda_gen_cf<- confusionMatrix( qda_gen_predict, test_data$Risk)
      qda_gen_cf
      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "QDA",
                Accuracy = qda_gen_cf$overall['Accuracy'] ,
                Sensitivity = qda_gen_cf$byClass['Sensitivity'],
                Specificity =  qda_gen_cf$byClass['Specificity'] )

# Support Vector Machines (SVM) Model
# -------------------------------------      
      svm_gen_model  <- svm(Risk~., data=train_data)
      svm_gen_predict <- predict(svm_gen_model, test_data)
      svm_gen_cf<- confusionMatrix( svm_gen_predict, test_data$Risk)
      svm_gen_cf
      df_gen_models <- 
        df_gen_models  %>% 
        add_row(Model = "SVM",
                Accuracy = svm_gen_cf$overall['Accuracy'] ,
                Sensitivity = svm_gen_cf$byClass['Sensitivity'],
                Specificity =  svm_gen_cf$byClass['Specificity'] )

      
# Summary of the Models we have build so far.
      df_gen_models[order(df_gen_models$Sensitivity,
                          df_gen_models$Specificity,decreasing = T),] %>% knitr::kable()


# Of the 8 models we have build KNN seems to be best option as of now as it has better balance between
# Sensitivity and Specificity value.
      

      
# Model Building using base methods from respective libraries
# -----------------------------------------------------------
        
# As mentioned in the begining of the file the models that we have build so far has been build with one fold (i.e splitting the dataset only once 
# into training and testing data. Now try multi fold / cross validation method by using "train" method 

        
# Similar to general model building exercise,  here in cross validation method also we shall finetune 
# those models for which important variables that made significant impact on the model building can be 
# identified and finetune the models based based on them and then select the one which provides best 
# result for the sensitivity/specificity pair

# Create data frame to store the output of different "train" models
      df_train_models <- data.frame(matrix(vector(),ncol=4))
      colnames(df_train_models) <- c("Model", "Accuracy", "Sensitivity", "Specificity")

# Control parameter for the train model
      ctrl <- trainControl( method="repeatedcv", repeats=5,	summaryFunction=twoClassSummary,	 
                 classProbs=TRUE,   allowParallel = TRUE)


# Logistic Regression Model
# -------------------------
      
# Building the Logistic Regression model
      glm_train_model <- train(Risk ~ ., method = "glm" ,  family = "binomial", 
                               data = train_data, trControl = ctrl)
      glm_train_predict <- predict(glm_train_model, test_data)
      glm_train_cf <- caret::confusionMatrix(glm_train_predict, test_data$Risk)

# List of Top 5 variables that made significant impact in Logistic Regression model
      glm_train_model_vi <- varImp(glm_train_model)
      glm_train_model_vi <- data.frame(feature   = rownames(glm_train_model_vi$importance),
                                       impact = glm_train_model_vi$importance$Overall)
      glm_train_model_vi[order(glm_train_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable()

# Finetuning the Logistic Regression model by the high impacting features
      glm_train_model_ft <- 
        train(Risk ~ Duration + checkingaccount + savingsaccounts + Sex + Purpose, 
              method = "glm" ,  family = "binomial", data = train_data, trControl = ctrl)
      glm_train_predict_ft <- predict(glm_train_model_ft, test_data)
      glm_train_cf_ft <- caret::confusionMatrix(glm_train_predict_ft, test_data$Risk)
      
      glm_train_cf_ft
      
      
# Between the vanila (i.e without any feature finetuning) & the finetuned model the sensitivity & 
# Specificity of the finetune model is higher. So we will short list that for comparing with other models

# Store the result to the train model dataframe
      df_train_models <- 
        df_train_models %>% 
        add_row(Model = "Logistic Regression_Train", 
                Accuracy = glm_train_cf_ft$overall['Accuracy'] , 
                Sensitivity = glm_train_cf_ft$byClass['Sensitivity'],
                Specificity =  glm_train_cf_ft$byClass['Specificity'])


# Decision Tree Model
# -------------------

# Building the Decision Tree
      dt_train_model <- train(Risk ~ ., method = "rpart" , data = train_data,  
                              metric="ROC",tuneGrid = expand.grid(.cp = seq(0.01,0.1,.01)), 
                              trControl=ctrl)

# Result of the Decision tree model for different CP values
      dt_train_model$results %>% dplyr::select(cp, Sens, Spec) %>% knitr::kable()

# Transforming the Decision tree train model output to required format for plotting the graph
      dt_train_model_tmp <- as.data.frame(dt_train_model$results) %>% dplyr::select(cp,Sens, Spec)
      dt_train_model_tmp <- gather(dt_train_model_tmp, Type, Val, Sens:Spec)
      dt_train_model_tmp <- dt_train_model_tmp [order(dt_train_model_tmp$cp ), ]

# Plotting the graph for Decision Tree model
      ggplot(dt_train_model_tmp, aes(cp, Val, color = Type)) + 
        geom_line() +
        geom_point(size = 3) + 
        scale_y_continuous(breaks = seq(0.10, 1.0, by = .05)) + 
        scale_x_continuous(breaks = seq(.01, 1.0, by = .01)) + 
        theme(panel.grid.minor.x = element_blank()) + 
        labs(y = "Sensitivity & Sepecificity", x = "Parameter - CP",
             title = "CART graph : Sensitivity & Sepecificity ")


# From the above graph plot we can see that for cp parameter of .01 we get optimum sensitivity/specificity
# pair values

# Data Record for the max Sensitivity value
      dt_train_model$results[dt_train_model$results$Sens == max(dt_train_model$results$Sens),] %>% 
        dplyr::select (cp, Sens, Spec)  %>% knitr::kable()

# Listing the TOP 5 variables that made significant impact Decision tree model
      dt_train_model_vi <- varImp(dt_train_model)
      dt_train_model_vi <- data.frame(feature   = rownames(dt_train_model_vi$importance), 
                                      impact = dt_train_model_vi$importance$Overall)
      
      dt_train_model_vi[order(dt_train_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable()

# Finetuning the Decision Tree model by the high impacting features
      dt_train_model_ft <- 
        train(Risk ~ creditamount + checkingaccount + savingsaccounts + Age + Purpose , 
              method = "rpart" , data = train_data,  metric="ROC",
              tuneGrid = expand.grid(.cp = seq(0.01,0.1,.01)), trControl=ctrl)

# Below is the OUtput for the finetuned version for differnt cp values
      dt_train_model_ft$results %>% dplyr::select(cp, Sens, Spec) %>% knitr::kable()

# Data record for the max Sensitivity value from the finetune model
      maxVal <- max(dt_train_model_ft$results$Sens)
      dt_train_model_ft$results[dt_train_model_ft$results$Sens == maxVal,] %>% 
        dplyr::select(cp, Sens, Spec)  %>% knitr::kable()



# Between the vanila and the finetune version of Decisition tree model we see that the sensitivity is 
# higher for the vanila model same in both the models the specificity is higher for the vanila version. 
# So we will short list the vanila version .By default the train model will pickup the data record with
# maximum ROC value.Since our requirement is to have the one with optimum Sensitivity + Specificity value
# pair we shall rebuild the model with the Cp value of the record with optimum sensitivity value


# cp parameter value for the data record with maximum sensitivity 
      cpVal = dt_train_model$results[dt_train_model$results$Sens == max(dt_train_model$results$Sens),]$cp 
      
      rm(dt_train_model)
      dt_train_model <- train(Risk ~ ., method = "rpart" , data = train_data,  
                              metric="ROC",tuneGrid = expand.grid(.cp = cpVal ), trControl=ctrl)
      dt_train_predict<- predict(dt_train_model, test_data)
      dt_train_cf<- caret::confusionMatrix(dt_train_predict, test_data$Risk)
      dt_train_model$results  %>% knitr::kable()
      df_train_models <- 
        df_train_models %>% 
        add_row( Model = "DT_Train", 
                 Accuracy = dt_train_cf$overall['Accuracy'] , 
                 Sensitivity = dt_train_cf$byClass['Sensitivity'],
                 Specificity =  dt_train_cf$byClass['Specificity'])


# Random Forest Model
# -------------------

# NOTE : Similar to the above RF base version this 'train' method version also took around 10 min 
#        for execution. If this is selected for production deployment then this performance factor has to be taken 
#        into consideration  
      
# Building the RF model for various values of mtry - the tuneGrid parameter
      rf_train_model <- train(Risk ~ ., method = "rf" , data = train_data, 
                              metric="ROC", tuneGrid = expand.grid(.mtry = seq(2,20,1)),
                              trControl=ctrl)

# Transforming the data frame to the structure to build the graph
      rf_train_model_tmp <- as.data.frame(rf_train_model$results) %>% dplyr::select(mtry,Sens, Spec)
      rf_train_model_tmp <- gather(rf_train_model_tmp, Type, Val, Sens:Spec)
      rf_train_model_tmp <- rf_train_model_tmp [order(rf_train_model_tmp$mtry ), ]

# Building the graph
      ggplot(rf_train_model_tmp, aes(mtry, Val, color = Type)) + 
        geom_line() +
        geom_point(size = 3) + 
        scale_y_continuous(breaks = seq(0.1, 1.0, by = .05)) + 
        scale_x_continuous(breaks = seq(2, 20, by = 2)) + 
        theme(panel.grid.minor.x = element_blank()) + 
        labs(y = "Sensitivity & Sepecificity", title = "mtry v/s Sensitivity & Sepecificity ")

# Data set of the record with maximum Sensitivity from the vanila model
      maxVal = max(rf_train_model$results$Sens)
      rf_train_model$results[rf_train_model$results$Sens == maxVal ,]  %>% knitr::kable()

# The TOP 5 features that had significant impact in the model outcome
      rf_train_model_vi <- varImp(rf_train_model)
      rf_train_model_vi <- data.frame(feature   = rownames(rf_train_model_vi$importance),
                                      impact = rf_train_model_vi$importance$Overall)
      rf_train_model_vi[order(rf_train_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable()

# Building the finetuned version of the Random forest method
      rf_train_model_ft <- train(Risk ~ creditamount + Age + Duration + Job + Sex, 
                                 method = "rf" , data = train_data, metric="ROC", 
                                 tuneGrid = expand.grid(.mtry = seq(2,20,1)), trControl=ctrl)

# Data set of the record with maximum Sensitivity from the finetuned model
      maxVal = max(rf_train_model_ft$results$Sens)
      rf_train_model_ft$results[rf_train_model_ft$results$Sens == maxVal,] %>% knitr::kable()


# Comparing the vanila & finetuned version , Sensitivity + Specificity pair is good for finetuned version. 
# Similar to the decision tree model case We shall re generate the Random Forest model also with the mtry
# value of finetuned method which has max Sensitivity value

      maxVal = max(rf_train_model_ft$results$Sens)
      mtryVal = rf_train_model_ft$results[rf_train_model_ft$results$Sens == maxVal,]$mtry
      rm(rf_train_model_ft)
      rf_train_model_ft <- train(Risk ~ creditamount + Age + Duration + Job + Sex, 
                                 method = "rf" , data = train_data, metric="ROC", 
                                 tuneGrid = expand.grid(.mtry = mtryVal), trControl=ctrl)
      rf_train_predict_ft <- predict(rf_train_model_ft, test_data)
      rf_train_cf_ft <- caret::confusionMatrix(rf_train_predict_ft, test_data$Risk)
      df_train_models <- 
        df_train_models %>% 
        add_row(Model = "Random Forest_Train", 
                Accuracy = rf_train_cf_ft$overall['Accuracy'] , 
                Sensitivity = rf_train_cf_ft$byClass['Sensitivity'],
                Specificity =  rf_train_cf_ft$byClass['Specificity'])



# Gradient Boosting Model
# -----------------------
      
# NOTE : This model took around 25 min for each of the execution. Take this performance factor into 
#        consideration should this be selected for production deployment 

# Building the graident boosting model
      xgb_train_model<- train(Risk ~ ., method = "xgbDART" , data = train_data)
      
      xgb_train_predict<- predict(xgb_train_model, test_data)
      xgb_train_cf<- confusionMatrix(xgb_train_predict, test_data$Risk)

# Listing  the TOP 5 variables that made significant impact in the Gradient boosting model
      xgb_train_model_vi <- varImp(xgb_train_model)
      xgb_train_model_vi <- data.frame(feature   = rownames(xgb_train_model_vi$importance),
                                       impact = xgb_train_model_vi$importance$Overall)
      xgb_train_model_vi[order(xgb_train_model_vi$impact,decreasing = T),][1:5,] %>% knitr::kable()

# Building the graident boosting model based on the high impacting features
      xgb_train_model_ft <- 
        train(Risk ~ Duration + checkingaccount + savingsaccounts + Sex + Purpose, 
              method = "xgbDART" , data = train_data)
      xgb_train_predict_ft <- predict(xgb_train_model_ft, test_data)
      xgb_train_cf_ft <- confusionMatrix(xgb_train_predict_ft, test_data$Risk)
      

# Comparing the vanilla and feature selection model of Gradient boosting we select the vanila model as 
# the Sensitivity/Specificity value pair is better than the feature selection version


# Adding the Gradient boosting result the to train model data frame
      df_train_models <<- 
        df_train_models %>% 
        add_row(Model = "Gradient Boosting_Train",
                Accuracy = xgb_train_cf$overall['Accuracy'] ,
                Sensitivity = xgb_train_cf$byClass['Sensitivity'],
                Specificity =  xgb_train_cf$byClass['Specificity'])
      

# KNN Model  
# ---------
        
# Building the KNN model
      knn_train_model<- train(Risk ~ ., method = "knn" , data = train_data, metric="ROC", 
                              tuneGrid = expand.grid(k = seq(1,9,1)),trControl = ctrl )

      knn_train_model


# Above details of the KNN model shows that sensitivity /specificity pair is significant when k is 3. 
# Though technically K=2 has slightly higher senisitivity than k=3, Specificity is significantly higher 
# for k=3 over K=2. Hence K=3 has been selected. As explained above the Model by default is considering 
# the value which has higher ROC and hence shows different value of K. As our requirement is to have 
# better Sensitivity  followed by specificity we will choose the value K=3 and rebuild the model

      rm(knn_train_model)
      knn_train_model <- train(Risk ~ ., method = "knn" , data = train_data, metric="ROC", 
                               tuneGrid = expand.grid(k = 3),trControl = ctrl )
      knn_train_predict<- predict(knn_train_model, test_data)
      knn_train_cf <- caret::confusionMatrix(knn_train_predict, test_data$Risk)


# Above shows the sensitivity and specificity details of the KNN method. This shows improvement over 
# other 'train' models though it is far below than the one we tried with general commands

# Adding the KNN result to the train model data frame
      df_train_models <- 
        df_train_models %>% 
        add_row(Model = "KNN_Train", 
                Accuracy = knn_train_cf$overall['Accuracy'] , 
                Sensitivity = knn_train_cf$byClass['Sensitivity'],
                Specificity =  knn_train_cf$byClass['Specificity'])

      
# Linear Discriminant Analysis (LDA)   
# ----------------------------------
# Building the LDA model
      lda_train_model <- train(Risk ~ ., method = "lda" , data = train_data,   trControl=ctrl)
      lda_train_predict<- predict(lda_train_model, test_data)
      lda_train_cf<- caret::confusionMatrix(lda_train_predict, test_data$Risk)
      lda_train_cf


# Above shows the sensitivity and specificity details of the LDA method. We see that this is far below 
# that what we have observed from other models 


# Adding the output to train model dataframe
      df_train_models <-  df_train_models %>% 
        add_row(Model = "LDA_Train", 
                Accuracy = lda_train_cf$overall['Accuracy'] , Sensitivity = lda_train_cf$byClass['Sensitivity'],
                Specificity =  lda_train_cf$byClass['Specificity'])
      

# Quadriatic Discriminant Analysis (QDA) model
# --------------------------------------------
        
# Building the QDA model
      qda_train_model<- 
        train(Risk ~ ., method = "qda" , data = train_data,  metric="ROC", trControl=ctrl)
      qda_train_predict<- predict(qda_train_model, test_data)
      qda_train_cf<- caret::confusionMatrix(qda_train_predict, test_data$Risk)
      qda_train_cf


# QDA Models' result is below the best one we have got so far
# Adding the QDA result to train model dataframe
      df_train_models <- 
          df_train_models %>% 
              add_row(Model = "QDA_Train", 
                      Accuracy = qda_train_cf$overall['Accuracy'] , 
                      Sensitivity = qda_train_cf$byClass['Sensitivity'],
                      Specificity =  qda_train_cf$byClass['Specificity'])


# Support Vector Machines (SVM) 
# -----------------------------
        
# Preparig the 2 finetuning parameters for the SVM method
      grid <- expand.grid(sigma = c(seq(.01, 0.10, 0.01)),
           C = c(seq(0.1, 1.0, .10)))

# Building the SVM train model
      svm_train_model <- train(Risk ~ ., method = "svmRadial" , 
                               data = train_data, tuneGrid = grid, trControl = ctrl)

# As it will be nearly 100 records we are not displaying the output of the svm train model. Instead we 
# shall check the out come with a graph

# Extracting only those values from the model output that are required to plot the graph
      svm_train_model_tmp <- as.data.frame(svm_train_model$results) %>% 
                           dplyr::select(sigma, C, Sens, Spec) 


# Now we shall plot the graph for SVM model. Since there are two different tuneGrid parameters of Sigma 
# and C we shall apply the formula of ( Sigma*100 + C ) which shall then be used to plot the graph

      svm_train_model_tmp <- svm_train_model_tmp %>%
         mutate(SigmaAndC = svm_train_model$results$sigma * 100 + svm_train_model$results$C) %>%
         dplyr::select(SigmaAndC, Sens, Spec)

#Transforming the data into another format in order to plot the graph
      svm_train_model_tmp <- gather(svm_train_model_tmp, Type, Val, Sens:Spec)

#Sorting it based on SigmaAndC value
      svm_train_model_tmp <- svm_train_model_tmp[order(svm_train_model_tmp$SigmaAndC ), ]

# Displaying only top 5 out of 200 values
      svm_train_model_tmp[1:5,] %>% knitr::kable()

# Plotting the graph
      ggplot(svm_train_model_tmp, aes(SigmaAndC, Val, color = Type)) +
         geom_line() +
         geom_point(size = 3) +
         scale_y_continuous(breaks = seq(0.10, 1.0, by = .06)) +
         scale_x_continuous(breaks = seq(1.00, 11.5, by = .60)) +
         theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
         labs(y = "Sensitivity & Sepecificity", x = "Parameter - Sigma And C",
         title = "SVN graph : Sensitivity & Sepecificity ")


# By default the model chooses the one with max ROC. As it is nearly 100 records we just displayed the 
# best value which the model will select by default
      svm_train_model$results %>% 
         filter(sigma == svm_train_model$bestTune$sigma, C == svm_train_model$bestTune$C )  %>% 
            knitr::kable()


# But our requirement is to select the one which has better combination of Sensitivity & Specificity. So 
# we shall display the top 5 results of maxium Sensitivity values
      svm_train_results <- data.frame(svm_train_model$results)
      svm_train_results[ order(svm_train_results$Sens,decreasing = T), ][1:5,] %>% 
         dplyr::select(sigma, C, Sens, Spec) %>% knitr::kable()

# We can see that the SVM model produce the best value when sigma = .01 and C = .3. Hence we will rebuild
# the model for that value


# selecting the tuning parameter of the record which yielded maximum sensitivity
      values <- data.frame(sigma,C)
      maxVal = max(svm_train_model$results$Sens)
      values <- svm_train_model$results[svm_train_model$results$Sens == maxVal,] %>% 
         dplyr::select(sigma,C)
      
      
      rm(svm_train_model)

# Finetuning the SVM model with parameters that yielded maximum Sensitivity value
      svm_train_model <- 
         train(Risk ~ ., method = "svmRadial" , data = train_data,
               tuneGrid = expand.grid(sigma= values$sigma, C = values$C), trControl = ctrl)
      svm_train_predict<- predict(svm_train_model, test_data)
      svm_train_cf<- caret::confusionMatrix(svm_train_predict, test_data$Risk)

# Confusion matrix Output of the SVM Train model
      svm_train_cf


# Result of the SVM model is found to be well below the maximum one we have observed so far

# Adding the SVM result to the train model dataframe
      df_train_models <- 
          df_train_models %>%
              add_row(Model = "SVM_Train",
                      Accuracy = svm_train_cf$overall['Accuracy'] ,
                      Sensitivity = svm_train_cf$byClass['Sensitivity'],
                      Specificity =  svm_train_cf$byClass['Specificity'])

# Comparing the train Models 
# --------------------------
      df_train_models %>%knitr::kable()


# Combining the output from both the general & Train models to figure out the right model
# ---------------------------------------------------------------------------------------
      df_combined_models <- rbind(df_train_models, df_gen_models)
      df_combined_models_tmp <- gather(df_combined_models, Type, Val, Sensitivity: Specificity)
      df_combined_models_tmp <- df_combined_models_tmp[order(df_combined_models_tmp[,1]),]
      df_combined_models_tmp$Model <- 
         factor(df_combined_models_tmp$Model, levels = unique(df_combined_models_tmp$Model))

# Plotting the graph for the combined list of model
# ---------------------------------------------------
      
      ggplot(df_combined_models_tmp, aes(Model, Val, color = Type, group = Type)) + 
         geom_line(aes(linetype = Type)) +
         geom_point(size = 3) + 
         theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
         labs(y = "Sensitivity & Sepecificity", x = "Models",
              title = "(General) Model comparision based on Sensitivity & Sepecificity ")


# Tabular display of the combined model result
# ---------------------------------------------

      df_combined_models[order(df_combined_models[,3], decreasing = T),]%>% knitr::kable()

