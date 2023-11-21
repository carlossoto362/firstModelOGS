\documentclass{article}
\usepackage{amsmath}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\usepackage{graphicx} % Required for inserting images
\usepackage[a4paper, total={6in, 8in}]{geometry}
\usepackage{fancyhdr}


\usepackage[backend=biber,
sorting=nyt,style=apa]{biblatex}
\addbibresource{bibliography.bib}

\usepackage{hyperref}

\title{NotesFirstModel}
\author{carlos.soto362@gmail.com }
\date{November 2023}

\begin{document}

\maketitle

\section{Introduction}

I will start with an informal explanation of the optical model, followed by a description of the inversion problem, and lastly a detailed description of the algorithm used. This document is meant to be an informal guide to my work, so please excuse me if there is something you don't understand. You can always contact me if you have any doubts. 

\section{Optical Model Description}

We are using a semi-empirical model for the optical dynamics between the incident irradiation on the sea, and the different components that scatter it or attenuate it in different ways, the so-called, optically active constituents. The model uses five constituents, which are seawater ($W$), phytoplankton ($PH$), colored dissolved organic matter ($CDOM$), and non-algal particles (NAP). The use of only five constituents is a really big assumption, for example, many types of phytoplankton have different scattering and absorption spectra, see \cite{dutkiewicz2015capturing}. Another big assumption made in the first model is that the phytoplankton and water constituents are constant in time.

How do these constituents interact with the incident radiation?
First, the incident radiation penetrates the seawater, due to the interaction in the interface between the Troposphere and the sea, the irradiation immediately up the interface and below it, is different. Later I will explain how this was handled. Next, the incident irradiation in a given direction, for each wavelength ($\lambda$), gets attenuated, due to the scattering and absorption of each constituent. The model follows only the downward scalar irradiance, so, if the direct radiance comes from a zenithal angle $\theta$, then, the downward scalar irradiance is 
\begin{equation}
    \nonumber E_{0dir} = E_{dir}/\cos{\theta},
\end{equation}
where $E_{0dir}$ is the downward scalar irradiance and $E_{dir}$ is the direct scalar irradiance. The model is a linear model (on the irradiance), which proposes that the change in the downward radiation for different depths, is proportional to the downward radiation on itself, with the proportionality factor given by the sum of two factors, one that represents the contributions given by the absorption($a_{\lambda}$), and other one for the scattering ($b_\lambda$), called total coefficients.

\begin{equation}
    \nonumber \frac{dE_{dir,\lambda}}{dz} = -\frac{a_\lambda + b_\lambda}{\cos{\theta}}E_{dir,\lambda}
\end{equation}

The radiation scattered is divided into radiation scattered downward ($E_{dif}$), and upward ($E_{u}$). Thus two are also been attenuated by scattering and absorption, in the same linear way as the direct radiation. The amount of radiation that is scattered in the inversed direction (upward for originally downward radiation and vice-versa) is equal to $r_db_{b,\lambda}$ for direct radiation, $r_ub_{b,\lambda}$ for upward radiation and $r_sb_{b,\lambda}$ for diffracted radiation; $r_d$, $r_u$ and $r_s$ are constants (values used from \cite{dutkiewicz2015capturing}). The equations for $E_{u}$ and $E_{dif}$ are

\begin{equation}
  \nonumber \frac{E_{dif,\lambda}}{dz} = -\frac{a_\lambda + r_sb_{b,\lambda}}{v_s}E_{dif,\lambda} + \frac{r_ub_{b,\lambda}}{v_u}E_{u,\lambda} + \frac{b_\lambda - r_db_{b,\lambda}}{\cos{\theta}}E_{dir,\lambda},
\end{equation}

\begin{equation}
  \nonumber \frac{E_{u,\lambda}}{dz} = -\frac{r_sb_{b,\lambda}}{v_s}E_{dif,\lambda} + \frac{a_\lambda + r_ub_{b,\lambda}}{v_u}E_{u,\lambda} - \frac{r_db_{b,\lambda}}{\cos{\theta}}E_{dir,\lambda}, 
\end{equation}
%
where $v_s$ and $v_u$ are the average direction cosines of the different irradiance streams, which are constants for diffused irradiance (values used from \cite{dutkiewicz2015capturing}). These equations need to follow the boundary conditions that the direct and diffuse irradiance at depth zero are equal to the ones given by the OASIM model, see \cite{lazzari2021cdom}, and that the upward radiation is zero at infinity depth. 

The total coefficients $a_\lambda$, $b_\lambda$ and $b_{b,\lambda}$ are computed as a linear combination of the contribution given by each one of the five constituents. 

\begin{equation}
    a_\lambda = a_{W,\lambda} + a_{PH,\lambda}chla + a_{CDOM,\lambda}CDOM + a_{NAP,\lambda}NAP 
    \label{eq:absortion}
\end{equation}
\begin{equation}
    b_\lambda = b_{W,\lambda} + b_{PH,\lambda}C + b_{NAP,\lambda}NAP 
\label{eq:scattering}
\end{equation}
\begin{equation}
    b_{b,\lambda} = b_{b,W,\lambda} + b_{b,PH,\lambda}C + b_{b,NAP,\lambda}NAP 
\label{eq:backscattering}
\end{equation}

with $a_{W,\lambda}$ (values used from \cite{pope1997absorption}), $a_{PH,\lambda}$ (values averaged and interpolated from \cite{alvarez2022phytoplankton}), $b_{W,\lambda}$ (values interpolated from \cite{smith1981optical},), $b_{PH,\lambda}$ (values used from \cite{dutkiewicz2015capturing}), $b_{b,W,\lambda}$ (scattering to backscattering ratio of 0.5 according to \cite{morel1974optical}) and $b_{b,PH,\lambda}$ (values used from \cite{dutkiewicz2015capturing}) constants,
\begin{equation}
    a_{CDOM,\lambda}=d_{CDOM}e^{-S_{CDOM}(\lambda - 450)}
    \label{eq:absortion_CDOM}
\end{equation}
$d_{CDOM}$ and $S_{CDOM}$ constants (\textbf{cite is missing}), 
\begin{equation}
    a_{NAP,\lambda}=d_{NAP}e^{-S_{NAP}(\lambda - 440)}
    \label{eq:absortion_NAP}
\end{equation}
$d_{NAP}$ and $S_{NAP}$ constants (equation and values used from \cite{gallegos2011long}), 
\begin{equation}
    b_{NAP,\lambda}=e_{NAP}\left(\frac{550}{\lambda}\right)^{f_{NAP}}
    \label{eq:scattering_NAP}
\end{equation}
$e_{NAP}$ and $f_{NAP}$ constants (equation and values used from \cite{gallegos2011long}), $b_{b,NAP,\lambda} = 0.005b_{NAP,\lambda}$, and
\begin{equation}
    C = chla/\left(\Theta_{chl}^0\frac{e^{-(PAR-\beta)/\sigma}}{1+e^{-(PAR-\beta)/\sigma}} + \Theta_{chl}^{min}\right)
    \label{eq:Carbon}
\end{equation}
$\Theta_{chl}^0$, $\sigma$, $\beta$, and $\Theta_{chl}^{min}$ constants (equation and values computed from \cite{cloern1995empirical}), and $PAR$ the Photosynthetically available radiation, obtained also from the OASIM model, see \cite{lazzari2021cdom}. 


For simplicity, the equations are written as

\begin{equation}
  \frac{dE_{dir,\lambda}}{dz} = -c_dE_{dir,\lambda},
  \label{eq:simplification}
\end{equation}
\begin{equation}
 \nonumber   \frac{dE_{dif,\lambda}}{dz} = -C_sE_{dif,\lambda} + B_uE_{u,\lambda} + F_dE_{dir,\lambda},
\end{equation}
\begin{equation}
 \nonumber   \frac{dE_{u,\lambda}}{dz} = -B_sE_{dif,\lambda}  + C_u E_{u,\lambda} - B_dE_{dir,\lambda},
\end{equation}

Using the assumption that none of the coefficients depend on the irradiance (an assumption that is not true, they depend at least on PAR, a function of the irradiances at different depths. ), this system of equations can be solved, with the solution,

\begin{equation}
    E_{dir,\lambda}(z)=E_{dir,\lambda}(0)e^{-zc_d}
    \label{eq:Edir}
\end{equation}
%
\begin{equation}
    E_{dif,\lambda}(z)=c^+e^{-k^+z}+xE_{dir,\lambda}(z)
    \label{eq:Edif}
\end{equation}
%
\begin{equation}
    E_{u,\lambda}(z)=c^+r^+e^{-k^+z}+yE_{dir,\lambda}(z)
    \label{eq:Eu}
\end{equation}

were,

\begin{equation}
    c^+=E_{dif,\lambda}(0) - xE_{dir,\lambda}(0)
    \label{eq:c+}
\end{equation}
\begin{equation}
    k^+=D-C_u
    \label{eq:k+}
\end{equation}
\begin{equation}
    r^+=\frac{B_s}{D}
    \label{eq:r^+}
\end{equation}
\begin{equation}
    D=\frac{1}{2}\left(C_s+C_u+\sqrt{\left(C_s + C_u \right)^2 - 4B_sB_u } \right)
    \label{eq:D}
\end{equation}
\begin{equation}
    x=\frac{1}{(c_d - C_s)(c_d + C_u)+B_sB_u}\left[ -(C_u+c_d)F_d - B_uB_d\right]
    \label{eq:x}
\end{equation}
\begin{equation}
    y=\frac{1}{(c_d - C_s)(c_d + C_u)+B_sB_u}\left[ -B_sF_d + (-C_s+c_d)B_d\right]
    \label{eq:y}
\end{equation}

\section{Inversion Problem}
Using the equations \ref{eq:Edir} to \ref{eq:Eu}, the optical model is a function, which gets a set of constants $\textbf{C}$ and a set of variables \newline $\chi = (E_{dif, \lambda}(0), E_{dir,\lambda}(0), \theta, PAR, chla, NAP, CDOM)$, and returns the different components for the irradiation, $\mathbf{E}_\lambda(z)=(E_{dir,\lambda}(z),E_{dif,\lambda}(z),E_{u,\lambda}(z))$,
\begin{equation}
\nonumber    \mathbf{E}_\lambda(z)=\mathbf{E}_\lambda(z;\textbf{C},\chi).
    \label{eq:model}
\end{equation}

This function can be related to a quantity that can be measured by satellite, the Remote Sensored Reflectance $R_{RS}$, by the relation,
\begin{equation}
    R_{RS}^{MODEL}=\frac{E_{u,\lambda}(0)}{Q(\theta)\left( E_{dir,\lambda}(0) + E_{dif,\lambda}(0) \right)}
    \label{eq:rrs}
\end{equation}
with,
\begin{equation}
    Q(\theta)= 5.33e^{-0.45\sin{(\pi/180(90-\theta))}}
    \label{eq:Q}
\end{equation}
equation from \cite{aas1999analysis}.

As mentioned before, the result obtained with this model would be the one seen below the interface between the seawater and the Atmosphere. An empirical solution is given by \cite{lee2002deriving}, with the relation, 
\begin{equation}
    R_{RS,down} = \frac{R_{RS,up} }{T + \gamma R_{RS,up}}
    \label{eq:rrsup}
\end{equation}
with $T$ and $\gamma$ constants. For simplicity, from now on, if $R_{RS}$ is mentioned, I'm going to assume that the required correction has been performed. In the same way as the irradiance, $R_{RS}$ depends on the wavelength, so, in most cases is going to be omitted, unless explicit dependence is required. 

Because $R_{RS}^{MODEL}$ can be obtain directly from the model $\mathbf{E}_\lambda(z;\textbf{C},\chi)$, then, 
\begin{equation}
  \nonumber  R_{RS}^{MODEL} = R_{RS,\lambda}^{MODEL}(\chi;\textbf{C})
\end{equation}

In terms of the availability of the data, the inversion problem divides the variables in two, $\chi_{0} = (E_{dif, \lambda}(0), E_{dir,\lambda}(0), \theta, PAR) $, and $\chi_{1}: \textbf{x} = (chla,CDOM,NAP)$.

Then, the inversion problem is described as, given the model,  $R_{RS,\lambda}^{MODEL}(\textbf{x};\chi_{0},\textbf{C})$, and the data measured by satellite $R_{RS,\lambda}^{OBS}$, obtained a prediction for $\textbf{x}_{p}$, minimizing a loss function, that describe the similarity between $R_{RS,\lambda}^{MODEL}(\textbf{x};\chi_{0},\textbf{C})$ and $R_{RS}^{OBS,\lambda}$.

\begin{equation}
\nonumber    \textbf{x}_{p} = \argmin_{\textbf{x}}{\left(L(\textbf{x};\chi_{0},\textbf{C},R_{RS,\lambda}^{OBS})\right)}.
\end{equation}

Currently, I'm using the Least Square Loss function, defined as
\begin{equation}
    L(\textbf{x};\chi_{0},\textbf{C},R_{RS,\lambda}^{OBS}) =\sum_\lambda \left( R_{RS,\lambda}^{MODEL}(\textbf{x};\chi_{0},\textbf{C}) - R_{RS,\lambda}^{OBS}\right)^2
    \label{eq:minimisation}
\end{equation}

\section{Code and Algorithm Description}
The algorithm is divided in two, the module with all the functions to read from files the constants, the data and the results, to define the model, and the functions to train the results, I called this module PySurfaceData. This module is the one that I'm supposed to leave untouched, in order to not mess something. The second file is called runingFirstModel.py, is the one I'm changing continuously, is the file that I use to run the model and store the results on files, and where I have many functions in order to plot the data. The code can be found in my \href{https://github.com/carlossoto362/firstModelOGS}{github}.

\subsection{PySurfaceData}
On this module I intend to store all the different models. Is composed of one \_\_init\_\_.py file, with a description of the module and the information of the author (me). 

For the moment, there is only one model, which is the one I already describe in the previous pages. The file that contains the model is called firstModel.py. On it, there is a function to read the constants needed, a function to reed the impute data for the model, and functions that compute each of the parts of the model. Starting with the equations needed in order to compute the  total coefficient (eq. \ref{eq:absortion}, \ref{eq:absortion_CDOM} and \ref{eq:absortion_NAP}), the equations for the scattering total coefficient (eq. \ref{eq:scattering}, \ref{eq:scattering_NAP} and \ref{eq:Carbon}), and the equation to compute the backscattering total coefficient (eq. \ref{eq:backscattering}). After this quantities are computed, there are the equations to compute $c_d$, $C_s$, $B_u$, $F_d$, $B_s$, $C_u$ and $B_d$ (eq. \ref{eq:simplification}), which are used to compute $c^+$ (eq. \ref{eq:c+}), $k^+$ (eq. \ref{eq:k+}), $r^+$ (eq. \ref{eq:r^+}), $D$ (eq. \ref{eq:D}), $x$ (eq. \ref{eq:x}) and $y$ (eq. \ref{eq:y}). All this, in order to compute the upward radiation on the surface (eq. \ref{eq:Eu}).

The final result for the model is obtained by computing the equations \ref{eq:Q}, \ref{eq:rrsup} and \ref{eq:rrs}, which results in $R_{RS}$.

$R_{RS}$ is a function of $E_{dif}(0)$,$E_{dir}(0)$,$\lambda$,$zenith$,$PAR$,$chla$,$NAP$ and $CDOM$. Using $R_{RS}$, a class is defined, named MODEL(torch.nn.Module), containing three torch.Parameter, which require gradients, this are the parameters that we want to learn, the chla, the NAP and the CDOM. The class also has a forward function, which is executed when an object of class MODEL is called. This function returns the evaluation of the model for each wavelength, using the current value of the parameters, which are initialised as a random number between zero and one.  In my first run, the data has five wavelengths, so, the forward model computes a six dimensional Tensor. 

Finally a train loop is defined, a function that executes the forward function upon the MODEL object, then evaluates a loss function between the evaluated MODEL and the real data, stops the values of the parameters to have negative values, compute the gradient of the loss function with respect of the parameters (the backward step), update the value of the parameters using an optimiser criterion, sets the gradient to zero, and store the result on a list. This procedure is perform $N$ times, where $N$ is a variable given to the function, as well as the data, the model, the loss function and the optimiser to used.

The output of these function is a list with the evaluation of the model for the different values of the parameters, and the value of the loss function at each iteration. 

\subsection{Running the Model}
The file runingFirstModel.py is the file that is used to run the Model, save the results, read the results from files, make plots of the results and get the statistics. The code consist of a header, where all the libraries and data required are loaded. After, there are two functions, which, in combination, using the model PySurfaceData, compute the parameters $chla$, $CDOM$ and $NAP$, as well as the value of $R_{RS}^{MODEL}$ and the value of the loss function. This process is paralleliced, for the moment, the learning of the parameters for one date is sequential, running on one single core, but the process of learning the parameters for several days is done in parallel.  

After storing the results, there are functions to read the results and add them to the pandas DataFrame where the data is, and functions to plot the data in different presentations. 


\section{Results}
For the moment, the model was used to performed the inversion procedure for all the data available from 2005 to 2012. Comparisons of the $R_{RS}^{MODEL}$ and $R_{RS}^{OBS}$ is shown in figure \ref{fig:rrs}, and \ref{fig:scatterRrs}.

Also using data from \textit{in situ} measurements was used to compare with the model data, results on figure \ref{fig:chl_com}. 

\begin{figure}
    \centering
    \includegraphics[width=0.6\textwidth]{chl_com.pdf}
    \caption{Comparison between $chla^{MODEL}$ and $chla^{OBS}$.}
    \label{fig:chl_com}
\end{figure}


\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{rrs_com.pdf}
    \caption{Comparison between $R_{RS}^{MODEL}$ and $R_{RS}^{OBS}$.}
    \label{fig:rrs}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{rrs_scattering.pdf}
    \caption{Scatter plot between $R_{RS}^{MODEL}$ and $R_{RS}^{OBS}$.}
    \label{fig:scatterRrs}
\end{figure}





\newpage
\printbibliography

\end{document}
