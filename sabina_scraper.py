from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import numpy as np
import xlrd
import os

"""
This python file contains a class called sabina_scraper which allows you to automatically download a beforehand specified Sabina table. 
Additionally the file includes functions to parse the downloaded files and load it into a pandas dataframe.
"""

def parse_xml(file, print_num_errors=False):
	"""
	Parses an xml file and returns the data as a pandas dataframe.

	Parameters
	----------
	file: Path of the file, which should be parsed
	print_num_errors: boolean, if True the number of parsing errors gets printed at the end

	Returns
	-------
	df: pandas dataframe with the parsed files
	"""
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    excounter = 0
    for child in root:
        tmp = {}
        for elem in child:
            try:
                # Use field+__+year as column header if year is available:
                tmp[elem.attrib["field"] + "__" + elem.attrib["year"]] = elem.text
            except:
                try:
                    # Otherwise use only field as column header:
                    tmp[elem.attrib["field"]] = elem.text
                except:
                    excounter += 1
                    pass
        data.append(tmp)

    df = pd.DataFrame(data)
    df = df.replace("n.v.", np.nan)

    if (print_num_errors):
        print(excounter)

    return df


def parse_folder_to_df(folder, datatype="xls"):
	"""
	Loads the xls/xml files from a specified folder, where the downloaded Sabina data is stored and returns a dataframe

	Parameters
	----------
	folder: path of the folder, where the downloaded Sabina data is stored
	datatype: type of the files in the folder, could be xls or xml

	Returns
	-------
	df_all: pandas dataframe containing the downloaded Sabina data
	"""
    dfs = []
    if datatype == "xml":
        files = glob.glob(f'./{folder}/sabina_*.{datatype}')
        for file in files:
            df = parse_xml(file)
            dfs.append(df)
        df_all = pd.concat(dfs)
    elif datatype == "xls":
        files = glob.glob(f"./{folder}/sabina_*.{datatype}")
        for file in files:
            wb = xlrd.open_workbook(file, logfile=open(os.devnull, 'w'))
            df = pd.read_excel(wb, engine='xlrd', sheet_name="Ergebnisse", index_col=0, encoding="utf-8")
            dfs.append(df)
        df_all = pd.concat(dfs)
    else:
        print("Not an acceptable datatype")
        df_all = None

    df_all.index = range(1, len(df_all) + 1)
    df_all = df_all.replace("n.v.", np.nan)

    return df_all

class sabina_scraper:
	"""
	This class allows you to automatically download data from the austrian balance database Sabina

	Parameters
	----------
	username: Your username of the Sabina online database
	pwd: Your password of the Sabina online database
	first_element: Integer index of the first element to download
	last_element: Integer index of the last element to download
	path_to_chromedriver: Enter the path to your chromedriver
	rows_per_download: number of rows included in each download except the last
	name_of_files: Name of the downloaded files

	"""

    def __init__(self,username = '...',
                      pwd = '...',
                      first_element = 0,
                      last_element = 278498,
                      path_to_chromedriver = "/usr/bin/chromedriver",
                      rows_per_download = 20000,
                      name_of_files = "sabina_table_export"):
        self.username = username
        self.pwd = pwd
        self.first_element = first_element
        self.last_element = last_element
        self.path_to_chromedriver = path_to_chromedriver
        self.rows_per_download = rows_per_download
        self.name_of_files = name_of_files



    def _start_webdriver_and_login_to_sabina(self):
	"""
	Starts the chromedriver and logs in into the Sabina online database. Afterwards the function 		checks for the autosave file, which should be downloaded.

	Returns
	-------
	The webdriver which has navigated to the autosave file from Sabina.
	"""
        # Using Chrome to access web
        driver = webdriver.Chrome("/usr/bin/chromedriver")
        # Open the website
        driver.get('https://sabina.bvdinfo.com/version-2019821/Home.serv?product=sabinaneo')
        # Select the id box
        id_box = driver.find_element_by_name("user")
        # Send id information
        id_box.send_keys(self.username)
        # Find password box
        pass_box = driver.find_element_by_name('pw')
        # Send password
        pass_box.send_keys(self.pwd)
        # Find login button
        login_button = driver.find_element_by_id('bnLoginNeo')
        # Click login
        login_button.click()
        try:
            driver.find_element_by_link_text("javascript:this.document.RestartForm.submit ()").click()
            print("Restart Clicked")
        except:
            pass

        # Click on Gespeicherte Suchen:
        gesp_suchen = driver.find_element_by_id("ContentContainer1_ctl00_Content_QuickSearch1_ctl02_TabSavedSearchesTd")
        gesp_suchen.click()
        not_found_autosave = True
        i = 0
        while (not_found_autosave):
            try:
                autosave = driver.find_element_by_id(
                    "ContentContainer1_ctl00_Content_QuickSearch1_ctl02_MySavedSearches1_DataGridResultViewer_ctl03_Linkbutton1")
                autosave.click()
                not_found_autosave = False
            except:
                time.sleep(1)
                i = i + 1
            if i == 10:
                break

        return driver

    def _start_data_download(self,driver, index):
	"""
	Downloads a shard of the autosave Sabina table

	Parameters
	----------
	driver: actual chromedriver
	index: starting index of the shard of the data, which should be downloaded
	"""
        timeout = 10
        # Go to initial Table page:
        driver.switch_to.window(driver.window_handles[0])

        # click export button:
        export = driver.find_element_by_id(
            "ContentContainer1_ctl00_Content_ListHeader_ListHeaderRightButtons_ExportButtons_ExportButton")
        export.click()
        settings_not_set = True
        
        while(settings_not_set):
            try:
                # Go to export settings:
                driver.switch_to.window(driver.window_handles[1])

                # insert export settings:

                # download range:
                start = driver.find_element_by_name("RANGEFROM")
                end = driver.find_element_by_name("RANGETO")
                start.clear()
                end.clear()
                start.send_keys(index * self.rows_per_download + 1)
                last_elem = (index + 1) * self.rows_per_download
                if (last_elem <= self.last_element):
                    end.send_keys((index + 1) * self.rows_per_download)
                else:
                    end.send_keys(self.last_element)

                # export format and download_name:
                data_type = Select(driver.find_element_by_id("exportformat"))
                #data_type.select_by_value("Xml")
                
                #export in xls format:
                data_type.select_by_value("ExcelDisplay2000")
                name = driver.find_element_by_id(
                    "ctl00_ContentContainer1_ctl00_LowerContent_Formatexportoptions1_ExportDisplayName")
                name.clear()
                name.send_keys(f"{self.name_of_files}_{index + 1}")

                # start download:
                download_button = driver.find_element_by_id("imgBnOk")
                download_button.click()
                settings_not_set = False
            except:
                time.sleep(timeout)

    def get_data(self):
	"""
	Starts the downloading process of the beforehand stored table from Sabina

	Parameters
	----------
	already initialized by creating the sabins_scraper instance

	Returns
	-------
	saves the downloaded Sabina table shards as xls files into your Downloads folder
	"""
        sabina_driver = self._start_webdriver_and_login_to_sabina()
        timeout = 10
        num_downloads = round(self.last_element / self.rows_per_download)
        print(num_downloads)
        i = 0
        while (i < num_downloads):
            try:
                print(f"Download {i} starts:")
                self._start_data_download(sabina_driver, i)
            except:
                print(f"Download {i} failed!")

            download_not_finished = True
            while (download_not_finished):
                try:
                    '''
                    search regurlarly for the exit button, when the button appears the download is finished
                    '''
                    download_finished = EC.presence_of_element_located(
                        (By.XPATH, '//*[@id="DownloadPanel"]/table/tbody/tr[2]/td/div/table/tbody/tr/td[2]/a'))
                    sabina_driver.find_element_by_xpath(
                        '//*[@id="DownloadPanel"]/table/tbody/tr[2]/td/div/table/tbody/tr/td[2]/a').click()
                    print(f"Download {i} finished!")
                    download_not_finished = False
                except:
                    time.sleep(timeout)
            i = i + 1

        print("All downloads from Sabina finished :)")


