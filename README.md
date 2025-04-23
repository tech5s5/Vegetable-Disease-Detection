# Vegetable Disease Detection

This project uses a combination of **React** for the frontend and **FastAPI** for the backend to provide a solution for detecting diseases in vegetables using machine learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Frontend (React)](#frontend-react)
- [Backend (FastAPI)](#backend-fastapi)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The **Vegetable Disease Detection** app allows users to upload images of vegetables, and it predicts the potential diseases that may be affecting the vegetables using a pre-trained machine learning model. The app is divided into two main parts:
- **Frontend**: Built with React, where users can upload images and view predictions.
- **Backend**: Powered by FastAPI, which handles the machine learning model inference and provides APIs for the frontend.

## Frontend (React)

The frontend is developed using React and allows users to:
- Upload images of vegetables for disease detection.
- View the predictions and related information.

### Key Features:
- Image upload functionality
- Displaying results from the backend API
- Responsive design for various screen sizes

## Backend (FastAPI)

The backend is built using **FastAPI** and serves the following purposes:
- Handles requests from the frontend for disease detection.
- Loads and runs the pre-trained machine learning model.
- Returns the disease predictions to the frontend.

The backend uses **TensorFlow** to load and use a pre-trained model to predict diseases from uploaded vegetable images.

### Key Features:
- Fast and efficient API with automatic validation.
- Model inference on uploaded images.
- Integration with the React frontend.

## Installation

### Prerequisites

- Python 3.7 or higher
- Node.js and npm (for React app)

### Steps to Set Up Backend

1. Clone the repository:
   ```bash
   git clone https://github.com/tech5s5/Vegetable-Disease-Detection.git
   cd Vegetable-Disease-Detection/api
