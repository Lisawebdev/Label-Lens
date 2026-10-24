# LabelLens – AI-Powered Dataset Label Analysis and Review System

LabelLens is a machine learning-based web application designed to analyse dataset labels and identify potentially incorrect or inconsistent records. It provides an interactive dashboard where users can upload CSV datasets, compare existing labels with machine learning predictions, review suspicious records, and export analysis results.

Built using Python, Flask, Pandas, Scikit-learn, HTML, CSS, and SQLite, LabelLens combines data processing, machine learning, and web development to support dataset quality assessment.

## Key Features
- **CSV Upload:** Upload datasets and select the target label column for analysis.
- **Automated Analysis:** Compare supplied labels with machine learning predictions to identify records that may require investigation.
- **Interactive Dashboard:** View dataset statistics, flagged records, and analysis results.
- **Record Review:** Mark records as Confirmed Incorrect, Valid Label, or Needs Review.
- **Review Notes:** Add comments to document observations and decisions.
- **Database Integration:** Store review statuses and notes for organised record management.
- **CSV Export:** Download analysis results for further examination.

## Technologies Used
Python, Flask, Pandas, Scikit-learn, HTML, CSS, and SQLite.

## Project Objective
The main objective of LabelLens is to simplify dataset inspection and provide a structured process for reviewing potentially incorrect labels. Rather than treating every prediction mismatch as a confirmed error, the application supports human verification before final review decisions are made.

LabelLens demonstrates the practical application of machine learning in data quality assessment and provides a foundation for future improvements, including advanced label detection techniques, additional models, enhanced visualisations, and more efficient processing of larger datasets.
