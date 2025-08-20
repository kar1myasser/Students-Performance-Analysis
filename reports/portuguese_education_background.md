# Portuguese Secondary Education System - Data Background Research

## Project Overview
This document provides contextual background for data analysis of Portuguese secondary education student performance, specifically focusing on the grading system and academic year structure that underlies the G1, G2, G3 variables in the dataset collected in 2008.

## Portuguese Secondary Education Structure

### Basic System Organization
Portuguese secondary education (ensino secundário) serves students aged 15-18 and consists of three grade levels:
- **10th Grade** (Décimo ano)
- **11th Grade** (Décimo primeiro ano) 
- **12th Grade** (Décimo segundo ano)

Students typically enter secondary education at age 15 after completing 9 years of basic education (ensino básico).

## Grading and Assessment System

### Grading Scale
Portuguese secondary education uses a standardized 20-point grading scale:
- **Scale Range**: 0-20 points
- **Passing Grade**: 10 or above
- **Excellent Performance**: 17-20 points
- **Good Performance**: 14-16 points
- **Satisfactory Performance**: 10-13 points

### Assessment Types
The Portuguese system employs three forms of assessment:

1. **Diagnostic Assessment**
   - Used for student integration and academic guidance
   - Strategic planning and course adjustments

2. **Formative Assessment**
   - Continuous and systematic evaluation
   - Regulatory function for teaching and learning processes
   - Helps identify and overcome difficulties

3. **Summative Assessment**
   - Primary grading and certification mechanism
   - Uses the 0-20 scale
   - Can be internal (school-based) or external (national exams)

### Academic Year Structure and Grading Periods

#### Understanding G1, G2, G3 Variables
The G1, G2, and G3 variables in the dataset represent **three assessment periods within a single academic year**, not different grade levels:

- **G1**: First period grade (1st trimester/term)
- **G2**: Second period grade (2nd trimester/term) 
- **G3**: Final year grade (3rd period/final assessment)

#### Temporal Structure
Portuguese academic years are typically organized into three periods:
- **1st Period**: September to December
- **2nd Period**: January to March/April
- **3rd Period**: April to June

Students receive formal grades at the end of each period, with G3 representing the final grade that determines academic progression.

<div style="page-break-before: always;"></div>

## Data Analysis Implications

### Data Leakage Considerations
The strong correlation between G1, G2, and G3 creates important methodological considerations:

**What's Happening:**
- G1 = 1st period grade (early in the year)
- G2 = 2nd period grade (middle of the year) 
- G3 = Final grade (end of year) - this is what we're trying to predict

The warning is saying that G1 and G2 are **highly correlated** with G3, meaning students who do well early in the year typically do well at the end too.

**The Data Leakage Problem:**
If we use G1 and G2 to predict G3, our model will be very accurate (high correlation = easy prediction), BUT this creates "data leakage" because:
- G2 happens chronologically very close to G3
- Using G2 to predict G3 is almost like cheating - we already know how the student performed late in the year

**Why This Matters for Early Intervention:**
The real educational value comes from predicting G3 using only:
- Student demographics
- Family background  
- Study habits
- School-related factors
- Maybe G1 (if we want some academic performance indicator)

**But NOT G2** - because by the time we have G2, it's too late for meaningful early intervention.

**The Trade-off:**
- **Easy but less useful**: Use G1 + G2 →

### Educational Context for Intervention
The Portuguese assessment structure supports early intervention strategies:
- G1 grades available by December/January provide early warning signals
- Students have multiple opportunities for improvement throughout the year
- The continuous assessment approach allows for ongoing support

<div style="page-break-before: always;"></div>

## System Stability and Historical Context

### 2008 Data Collection Period
The dataset was collected in 2008, during a period when the Portuguese education system had already established its current basic structure:

- The Education Act of 1986 established the fundamental framework
- The 0-20 grading scale was standard practice
- Multiple assessment periods per year were common practice
- Secondary education structure (grades 10-12) was well-established

### Recent Reforms (Post-2008)
While the basic structure remains consistent, notable changes since 2008 include:
- Decree-Law No. 55/2018: Included physical education grades in final averages
- Decree-Law No. 62/2023: Modified national exam requirements
- Enhanced flexibility in course pathway selection

## University Admission Context

### National Exams
Students must take national standardized exams in:
- Portuguese (mandatory for all students)
- Two additional subjects from their specialization area
- These typically occur in 11th and 12th grades

### Admission Calculation
University entrance scores combine:
- Overall secondary education grades (includes all three years)
- Specific national exam scores
- Different weightings by program and 

<div style="page-break-before: always;"></div>

## Sources and References

### Primary Sources
1. **[Eurydice Network](https://eurydice.eacea.ec.europa.eu/national-education-systems/portugal/assessment-general-upper-secondary-education)** (Official EU Education Documentation)
   - Description: Comprehensive official documentation of Portuguese assessment practices
   - Key Evidence: "Students are internally assessed at the end of each term/semester and school year"

2. **[Wikipedia - Academic Grading in Portugal](https://en.wikipedia.org/wiki/Academic_grading_in_Portugal)**
   - Confirmation of 0-20 grading scale usage)

### Supporting Sources
3. **[Portuguese Directorate-General for Education](https://www.dge.mec.pt/)**
   - Official guidance on diploma and certificate issuance
   - Assessment criteria and progression requirements

## Limitations and Considerations

### Source Limitations
- Limited access to specific 2008 Portuguese Ministry of Education documents
- Most detailed sources reflect current system (2018-2025)
- Assumption that basic assessment structure remained consistent from 2008 to present

### Data Analysis Considerations
- G1, G2, G3 represent within-year progression, not across grade levels
- Strong temporal correlation requires careful handling in predictive models
- Cultural and institutional context may affect generalizability to other education systems

## Research Implications

This background research supports the understanding that the dataset captures student performance across three assessment periods within a single academic year in the Portuguese secondary education system, using the standardized 0-20 grading scale that has been consistent in Portuguese education for decades.