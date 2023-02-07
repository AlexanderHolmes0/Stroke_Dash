library(shiny)
library(tidyr)
library(caret)
library(ggplot2)
library(dplyr)
library(pROC)
library(shinydashboard)
options(warn=-1)#stop the pesky warnings of caret while training the model

stroke_data <- read.csv("healthcare-dataset-stroke-data.csv", 
                        stringsAsFactors = T)

#clean up the data a lil bit making names more 'r' friendly
stroke_data <- stroke_data %>%
  mutate(hypertension = factor(hypertension),
         heart_disease = factor(heart_disease),
         stroke = factor(stroke, labels = c("no","yes")),
         smoking_status = recode_factor(smoking_status, 'never smoked'="never_smoked", 'formerly smoked'='formerly_smoked'),
         work_type = recode_factor(work_type, 'Self-employed'='Self_employed'),
         bmi =suppressWarnings(as.numeric(as.character(bmi)))) %>% 
  replace_na(list(bmi = 0)) %>% #change those NAs to zeros, theres only 200 of em
  select(-c(id)) #deselect that id column to make life easier

stroke_data <- stroke_data[sample(1:nrow(stroke_data)),]



ui <- dashboardPage(
  dashboardHeader(title = "Stroke Prediction"),
  
  dashboardSidebar(
      sidebarMenu(id = "sidebar",
        menuItem("Health Information", tabName = "home" ,icon = icon("suitcase-medical"),
                 badgeLabel = "new", badgeColor = "green"),
        
        menuItem("Specifics", icon=icon("microscope"),tabName = "specifi",
                 badgeLabel = "new", badgeColor = "green"),
        
        menuItem("Holdout Proportion", icon=icon("bar-chart"),tabName = "prop",
                 badgeLabel = "new", badgeColor = "green"),
        
        menuItem("Source code", icon = icon("github"), 
                 href = "https://github.com/AlexanderHolmes0/Stroke_Dash",
                 badgeLabel = "new", badgeColor = "green"),
        
        conditionalPanel(
          condition = "input.sidebar == 'prop'",
          sliderInput("mix",
                      h4("Training Data Proportion (How Much the Model Knows)"),
                      min = 0.01,
                      max = 0.99,
                      value = .6)),
        
        conditionalPanel('input.sidebar == "home"',
          sliderInput("Age",
                      h4("Select Age:"),
                      min = 0,
                      max = 120,
                      value = 20),
          sliderInput("BMI",
                      h4("Select BMI:"),
                      min = 0,
                      max = 60,
                      value = 20),
          
          
          selectInput("Gender", label = h4("Select Gender"), 
                      choices = c("Female" = "Female", "Male" = "Male", "Other" = "Other")
          ),
          
          selectInput("Hypertension", label = h4("Hypertension?"), 
                      choices = c("Yes" = "1","No"="0"),selected = "0" #selected argument needs to match what the selection actually represents 
                      #i.e. the data is in 0 or 1 so we "select" the 0 not what the choice in the input that represents that selection.
          ),
          
          selectInput("Heart", label = h4("Heart Disease?"), 
                      choices = c("Yes" = "1","No"="0"),selected = "0"
          ),
         
          
          selectInput("Smoke", label = h4("Smoking Status"), 
                      choices = c("Never Smoked" = "never_smoked",
                                  "Formerly Smoked" = "formerly_smoked",
                                  "Smokes" = "smokes",
                                  "Unknown (Really Now?)" = "Unknown"
                      ),
                      selected = "Never Smoked"
          )
        ),
        conditionalPanel(
          "input.sidebar == 'specifi'",
          sliderInput(
            "Avg_glucose",
            h4("Average Glucose Levels:"),
            min = 30,
            max = 300,
            value = 100
          ),
          selectInput(
            "Ever_Married",
            label = h4("Married"),
            choices = c("Yes" = "Yes", "No" = "No"),
            selected = "No"
          ),
          selectInput(
            "Work_Type",
            label = h4("Work Type"),
            choices = c(
              "Self Employed" = "Self_employed",
              "Children" = "children",
              "Government" = "Govt_job",
              "Never Worked" = "Never_worked",
              "Private" = "Private"
            ),
            selected = "Self Employed"
          ),
          selectInput("Residence_Type", label = h4("Residence Type"), 
                      choices = c("Rural (Country)" = "Rural",
                                  "Urban (City)" = "Urban"
                      ),
                      selected = "Rural (Country)"
          )
        )
        
      )), 
   
  
  
  dashboardBody(
    
    tags$head(tags$link(rel = "shortcut icon", href = "favicon.ico")),
    
    fluidRow(
      tabBox(
        id = "tabset1", height = "1000px",width="1000px",
        tabPanel("Probability",valueBoxOutput("StrokeProb")),
        tabPanel("Holdout Preds", verbatimTextOutput("Label1"),
                 plotOutput("Sigmodial")),
        tabPanel("AUROC", verbatimTextOutput("ROCLabel"),
                 plotOutput("ROC")),
        tabPanel("Density", verbatimTextOutput("label3"),plotOutput("Density"))
        )
      )
  )
)




server <- function(input, output) {

    #store objects that need to change in reactive() objects, acce$$ed later...
    train.rows <- reactive({
      sample(1:nrow(stroke_data), input$mix * nrow(stroke_data))
    })
    
    TRAIN <- reactive({
      stroke_data[train.rows(), ]
    })
    
    HOLDOUT <- reactive({
      stroke_data[-(train.rows()), ]
    })
    
    logi <-
      reactive({
        #remember that reactive() objects are just like functions
        fitControl <-
          trainControl(method = "boot",
                       number = 5,
                       classProbs = TRUE)
        
        train(stroke ~ .,
              data = TRAIN(),
              method = "glm",
              trControl = fitControl)
        
        
      })
    
    preds <- reactive({
      
      #return the second column of the predictions as those are the YES class predictions
      
      data.frame(Stroke_Probability = as.numeric(predict(logi(), HOLDOUT() , type =
                                                           "prob")[, 2]),
                 #Nifty use of Reactive() obj acce$$ing vars
                 Age = HOLDOUT()$age,
                 Stroke = HOLDOUT()$stroke,
                 BMI = HOLDOUT()$bmi
      )
    }
    )
    
    #User can create an individual and get a generated probability from the vanilla logistic model
    output$StrokeProb <- renderValueBox({
      Input <- data.frame(gender = input$Gender, age = input$Age,
                         hypertension = input$Hypertension, 
                         heart_disease = input$Heart, 
                         ever_married = input$Ever_Married, 
                         work_type = input$Work_Type,
                         Residence_type = input$Residence_Type, 
                         avg_glucose_level = input$Avg_glucose, 
                         bmi=input$BMI, 
                         smoking_status=input$Smoke)
      valueBox(
        paste(round(predict(logi(), Input, type="prob")[2],2)*100, "%", sep=""),"Chance of Stroke",
        color = "red"
      )
    })
    
    
    
    
    
    
    #keep stuff clean and let user know what things are
    output$Label1 <- renderPrint(
      
      "Predicted probabilites of stroke in individuals within the Test (Holdout) dataset."
    )
    
    #This is pretty complicated to the untrained eye but remember that reactive() objects are just functions that return something
    output$Sigmodial <- renderPlot({
      
      
      ggplot(preds(), aes(x = Age, y = Stroke_Probability)) +
        geom_point() +
        labs(x = "Age (Years)", y = "Predicted Stroke Probability (YES Class)") +
        ggtitle("Age vs. Model Stroke Probability") +
        geom_smooth(color = 'green') +
        scale_y_continuous(labels = scales::percent) +
        theme_bw() +
        theme(text = element_text(size = 15),
              plot.title = element_text(hjust = 0.5))
      
      
    })
    
    output$ROC <- renderPlot({
      
      
      roc_full <- roc(preds()$Stroke,  preds()$Stroke_Probability )
      auc <- round(auc(preds()$Stroke, preds()$Stroke_Probability),4)
      
      ggroc(roc_full,size=1) +
        geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="green", linetype="dashed")+
        ggtitle(paste0('ROC Curve ', '(AUC = ', auc, ')')) +
        labs(x= "Specificity (False Positives: Bad)",
             y= "Sensitivity (True Positives: Good)")+
        theme_bw()+
        theme(text = element_text(size = 15),
              plot.title = element_text(hjust = 0.5))
      
    })
    
    
    output$ROCLabel <- renderPrint(
      
      "AUROC of the stroke model."
    )
    
    output$Density <- renderPlot({
      
      ggplot(preds(), aes(x = Age, color=Stroke, fill=Stroke)) +
        geom_density(alpha=.5) +
        labs(x = "Age (Years)",y="Density") +
        ggtitle("Age vs. Model Stroke Probability") +
        
        theme_bw() +
        theme(text = element_text(size = 15),
              plot.title = element_text(hjust = 0.5))
      
      
    })
    
    
    output$label3 <- renderPrint({
      
      "The Density of the stroke predictions of the model by age"
      
      
    })
    
    
  }
  
  
  
  


shinyApp(ui, server)
