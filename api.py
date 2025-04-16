from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from typing import List, Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup
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
from fastapi.concurrency import run_in_threadpool

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


# ---------------------- Async API Endpoints ----------------------

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Scraper API!"}


@app.post("/scrapeSearch/")
async def scrape_google_search(request: ScrapeRequest):
    try:
        raw_html = await run_in_threadpool(fetch_html_selenium, request.url)
        soup = BeautifulSoup(raw_html, 'html.parser')

        seen = set()
        results_list = []

        results = soup.find_all('div', class_='CA5RN')

        for item in results:
            title_tag = item.find('span', class_='VuuXrf')
            cite_tag = item.find('cite')

            if title_tag and cite_tag:
                title = title_tag.text.strip()
                link = cite_tag.get_text(strip=True)
                if (title, link) not in seen:
                    seen.add((title, link))
                    results_list.append({"title": title, "link": link})

        return results_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeLinks/")
async def scrape_links(request: ScrapeRequest):
    try:
        raw_html = await run_in_threadpool(fetch_html_selenium, request.url)
        internal_links = extract_internal_links_from_html(raw_html, request.url)
        data = await run_in_threadpool(extract_links, internal_links, request.selected_model)
        return {"urls": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeData/")
async def scrape_data(request: ScrapeRequest):
    try:
        raw_html = await run_in_threadpool(fetch_html_selenium, request.url)
        data = extract_data(raw_html, {"type": "home", "url": request.url})
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeMultiple/")
async def scrape_multiple_data(request: MultiScrapeRequest):
    try:
        return await run_in_threadpool(run_bulk_scraper, request.urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeAllData/")
async def scrape_all_data(request: ScrapeRequest):
    try:
        results = []

        raw_html = await run_in_threadpool(fetch_html_selenium, request.url)
        base_data = extract_data(raw_html, {"type": "home", "url": request.url})
        results.append({"url": request.url, "data": base_data})

        internal_links = extract_internal_links_from_html(raw_html, request.url)
        relevant_links = await run_in_threadpool(extract_links, list(internal_links), request.selected_model)
        internal_results = run_bulk_scraper(relevant_links)
        results.extend(internal_results["results"])

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrapeLinkedInAds/")
async def scrape_linkedin_ads(request: LinkedInRequest):
    try:
        base_url = 'https://www.linkedin.com/ad-library/search'
        params = {
            'accountOwner': request.account,
            'countries': request.country,
            'dateOption': request.date
        }
        url = f"{base_url}?{urlencode(params)}"

        raw_html = await run_in_threadpool(fetch_html_selenium, url)
        data = extract_linkedIn_ads_data(raw_html)

        return {"linkedin_ads": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")
