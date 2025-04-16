from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from typing import List, Dict, Optional
import os
import re
import time
import random
import json
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlencode

import pandas as pd
from bs4 import BeautifulSoup
import html2text
import tiktoken
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as gena
from groq import Groq
from scraper import (
    fetch_html_selenium,
    extract_links,
    extract_data,
    run_bulk_scraper,
    extract_internal_links_from_html,
    extract_linkedIn_ads_data,
)

load_dotenv()

app = FastAPI()

# ---------------------- Pydantic Models ----------------------

class ScrapeRequest(BaseModel):
    url: str
    selected_model: Optional[str]

class MultiScrapeRequest(BaseModel):
    urls: List[str]

class LinkedInRequest(BaseModel):
    account: str
    country: str
    date: str

# ---------------------- API Endpoints ----------------------

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Scraper API!"}


@app.post("/scrapeSearch/")
def scrape_googleSearch(request: ScrapeRequest):
    try:
        raw_html = fetch_html_selenium(request.url)
        print("raw_html", raw_html)
        soup = BeautifulSoup(raw_html, 'html.parser')

        # Set to avoid duplicates
        seen = set()
        results_list = []

        # Find all result blocks
        results = soup.find_all('div', class_='CA5RN')
        print("results", results)

        for item in results:
            title_tag = item.find('span', class_='VuuXrf')
            cite_tag = item.find('cite')

            if title_tag and cite_tag:
                title = title_tag.text.strip()
                link = cite_tag.get_text(strip=True)

                # Use a tuple to check for duplicates
                if (title, link) not in seen:
                    seen.add((title, link))
                    results_list.append({
                        "title": title,
                        "link": link
                    })

        # Output JSON
        json_output = json.dumps(results_list, indent=2, ensure_ascii=False)
        print(json_output)

    except Exception as e:
        return {"detail": str(e)}


@app.post("/scrapeLinks/")
def scrape_links(request: ScrapeRequest):
    """Scrape a URL and return structured data."""
    try:
        url = request.url
        raw_html = fetch_html_selenium(url)
        internal_links = extract_internal_links_from_html(raw_html, url)
    
        data = extract_links(internal_links, request.selected_model)
        
        return  {"urls": data }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrapeData/")
def scrape_data(request: ScrapeRequest):
    """Scrape a URL and return structured data."""
    try:
        url = request.url
        raw_html = fetch_html_selenium(url)

        data = extract_data(raw_html, {"type": "home", "url": url})

        return {
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/scrapeMultiple/")
def scrape_multiple_data(request: MultiScrapeRequest):  
    urls = request.urls
    return run_bulk_scraper(urls)

@app.post("/scrapeAllData/")
def scrape_all_data(request: ScrapeRequest):
    try:
        results = []

        url = request.url
        raw_html = fetch_html_selenium(url)
        base_data = extract_data(raw_html, {"type": "home", "url": url})
        results.append({"url": url, "data": base_data})

        internal_links = extract_internal_links_from_html(raw_html, url)
        print(f"🔗 Extracted internal links: {len(internal_links)}")

        relevant_links = extract_links(list(internal_links), request.selected_model)
        print(f"✅ Filtered relevant links: {len(relevant_links)}")

        internal_results = run_bulk_scraper(relevant_links)
        results.extend(internal_results["results"])

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeLinkedInAds/")
def scrape_linkedin_ads(request: LinkedInRequest) :
    """Scrape LinkedIn ad library and return structured ad data."""
    try:
        base_url = 'https://www.linkedin.com/ad-library/search'
        params = {
            'accountOwner': request.account,
            'countries': request.country,
            'dateOption': request.date
        }
        url = f"{base_url}?{urlencode(params)}"

        raw_html = fetch_html_selenium(url)  # assumes you have a Selenium scraper
        data = extract_linkedIn_ads_data(raw_html)

        return {
            "linkedin_ads": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")
    
