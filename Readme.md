## The Imputation toolkit
The toolkit contains functions to test different imputation strategies on a dataset. Additionally there are some useful helper functions to visualize the percentage of missing values per column, as well as the overall imputation result in form of Qplot.

### Preconditions for the toolkit:
> pip install sklearn
> pip install pandas
> pip install matplotlib
> pip install numpy
> pip install tqdm

### Tutorial:
The imputation_toolkit_sabina.ipynb notebook shows a step by step solution how to define your imputation methods and how you run the testing process.

### Sabina Scraper:
This project was originally designed to automate the download process of the austrian company balance data from [https://sabina.bvdinfo.com/version-2019821/Login.serv?Code=AccountExpired&LoginParamsCleared=True&LoginResult=nc&product=sabinaneo&RequestPath=home.serv%3fproduct%3dSabinaNeo][Sabina] and afterwards impute the missing values.
The sabina_scraper file contains the sabina_scraper class which can be used for this downloading process. The imputation toolkit itself is independent from the Sabina database and can be used for any dataset.

### Preconditions for the sabina scraper:
>pip install selenium
>install a [https://chromedriver.chromium.org/][chromedriver]


