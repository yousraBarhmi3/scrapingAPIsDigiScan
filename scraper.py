import os
import random
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type
from urllib.parse import urlparse, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from openai import OpenAI
import google.generativeai as genai
from groq import Groq
from webdriver_manager.chrome import ChromeDriverManager


from api_management import get_api_key
from assets import USER_AGENTS,HEADLESS_OPTIONS,SYSTEM_MESSAGE,evaluation_message,USER_MESSAGE,LLAMA_MODEL_FULLNAME,GROQ_LLAMA_MODEL_FULLNAME,HEADLESS_OPTIONS_DOCKER, LINKS_MESSAGE
load_dotenv()


# Set up the Chrome WebDriver options
import shutil
import subprocess


def is_running_in_docker():
    """
    Detect if the app is running inside a Docker container.
    This checks if the '/proc/1/cgroup' file contains 'docker'.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception:
        return False

def setup_selenium(attended_mode=False):
    print("🧪 DEBUG: chrome version ->", subprocess.getoutput("google-chrome --version"))
    print("🧪 DEBUG: chromedriver version ->", subprocess.getoutput("chromedriver --version"))
    options = Options()
    for option in HEADLESS_OPTIONS_DOCKER:
        options.add_argument(option)

    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.binary_location = "/opt/chrome/chrome"
    service = Service(ChromeDriverManager().install())

    # service = Service(executable_path="/usr/local/bin/chromedriver")

    driver = webdriver.Chrome(service=service, options=options)
    return driver


def fetch_html_selenium(url, attended_mode=False, driver=None):
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
        if not attended_mode:
            driver.get(url)
    else:
        should_quit = False
        # Do not navigate to the URL if in attended mode and driver is already initialized
        if not attended_mode:
            driver.get(url)

    try:
        if not attended_mode:
            # Add more realistic actions like scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.3);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/0.8);")
            time.sleep(random.uniform(1.1, 1.8))
        # Get the page source from the current page
        html = driver.page_source
        return html
    finally:
        if should_quit:
            driver.quit()

def extract_internal_links_from_html(raw_html, url):
    soup = BeautifulSoup(raw_html, "html.parser")

    base_domain = urlparse(url).netloc
    normalized_url = url.rstrip('/')

    links = soup.find_all('a', href=True)
    
    internal_links = {
        urljoin(url, a['href']).rstrip('/')
        for a in links
        if urlparse(urljoin(url, a['href'])).netloc == base_domain
        and urljoin(url, a['href']).rstrip('/') != normalized_url
    }

    return list(internal_links)


def extract_links(links: List[str], selected_model: str) -> List[str]:
    user_prompt = f"{USER_MESSAGE} {json.dumps(links, ensure_ascii=False)}"

    def parse_response(content):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Parsed content is not a list.")
        except json.JSONDecodeError:
            raise ValueError("Model response is not valid JSON.")

    if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
        client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": LINKS_MESSAGE},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        return parse_response(content)

    elif selected_model == "gemini-1.5-flash":
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        )
        response = model.generate_content(f"{LINKS_MESSAGE}\n{USER_MESSAGE} {json.dumps(links)}")
        content = response.candidates[0].content.parts[0].text
        return parse_response(content)

    elif selected_model == "Groq Llama3.1 70b":
        client = Groq(api_key=get_api_key("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_LLAMA_MODEL_FULLNAME,
            messages=[
                {"role": "system", "content": LINKS_MESSAGE},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.choices[0].message.content
        return parse_response(content)

    else:
        raise ValueError(f"Unsupported model: {selected_model}")
    
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

def extract_data(html_content, url):
    soup = BeautifulSoup(html_content, "lxml")
    
    # === A. Structure & SEO
    title = soup.title.string.strip() if soup.title else ""
    meta_description = soup.find("meta", attrs={"name": "description"})
    description = meta_description.get("content") if meta_description else ""

    robots = soup.find("meta", attrs={"name": "robots"})
    meta_robots = robots.get("content") if robots else ""

    canonical = soup.find("link", rel="canonical")
    canonical_url = canonical.get("href") if canonical else ""

    og_tags = {tag.get("property"): tag.get("content") for tag in soup.find_all("meta") if tag.get("property", "").startswith("og:")}
    twitter_tags = {tag.get("name"): tag.get("content") for tag in soup.find_all("meta") if tag.get("name", "").startswith("twitter:")}

    html_lang = soup.html.get("lang") if soup.html and soup.html.has_attr("lang") else ""

    charset = soup.find("meta", attrs={"charset": True})
    viewport = soup.find("meta", attrs={"name": "viewport"})
    meta_charset = charset.get("charset") if charset else ""
    meta_viewport = viewport.get("content") if viewport else ""

    headings = {
        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")]
    }

    # === B. Contenu Principal & Pertinence
    all_text_tags = soup.find_all(["p", "li", "span"])
    full_text = " ".join([tag.get_text(strip=True) for tag in all_text_tags])
    word_count = len(full_text.split())

    # === C. UX / Conversion / CTA
    buttons = soup.find_all(["a", "button"])
    ctas = []

    for btn in buttons:
        text = btn.get_text(strip=True)
        href = btn.get("href")
        ctas.append({
            "text": text,
            "href": urljoin(url, href) if href else None,
        })

    # === D. Accessibilité & Image
    images = soup.find_all("img")
    img_data = [{
        "src": urljoin(url, img.get("src")),
        "alt": img.get("alt", "")
    } for img in images if img.get("src")]

    text_vs_images = {
        "total_text_length": len(full_text),
        "total_images": len(images),
        "ratio_text_per_image": len(full_text) / len(images) if images else "∞"
    }

    # === E. Liens internes & navigation
    links = soup.find_all("a", href=True)
    internal_links = [{
        "href": urljoin(url, a["href"]),
        "text": a.get_text(strip=True)
    } for a in links if a["href"].startswith("/") or url in a["href"]]

    key_pages = [l for l in internal_links if re.search(r"/(contact|devis|espace|produit|services|simulateur)", l["href"], re.IGNORECASE)]

    # === F. Réseaux sociaux
    social_platforms = ["facebook.com", "instagram.com", "linkedin.com", "tiktok.com", "youtube.com", "x.com", "twitter.com"]
    social_links = [{
        "href": a["href"],
        "icon_alt": a.get("aria-label") or a.get("title") or a.get("alt", "")
    } for a in links if any(social in a["href"] for social in social_platforms)]

    return {
        "seo": {
            "title": title,
            "description": description,
            "lang": html_lang,
            "meta_charset": meta_charset,
            "meta_viewport": meta_viewport,
            "meta_robots": meta_robots,
            "canonical": canonical_url,
            "og_tags": og_tags,
            "twitter_tags": twitter_tags,
            "headings": headings
        },
        "content": {
            "word_count": word_count,
            "text_sample": full_text[:300] + "..." if len(full_text) > 300 else full_text
        },
        "cta_analysis": {
            "cta_links": ctas
        },
        "images": {
            "all_images": img_data,
            "text_vs_images": text_vs_images
        },
        "navigation": {
            "internal_links": internal_links,
            "key_pages": key_pages
        },
        "social_presence": {
            "detected_links": social_links,
            "total_socials": len(social_links)
        }
    }

def evaluate(data):

    # Configure the Google Gemini API
    GEMINI_API_KEY = 'AIzaSyA2WbooEs3dMYVOy2PDEe27QstNsPwE62s'

    # Configurer le client Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    # genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
    # Initialize the model
    model = genai.GenerativeModel("gemini-1.5-flash")
    # Construct the prompt
    prompt = evaluation_message + "\n" + USER_MESSAGE + data

    # Generate and parse the response
    try:
        completion = model.generate_content(prompt)
        response_text = completion.candidates[0].content.parts[0].text
        return response_text  # Return parsed JSON response
    except Exception as e:
        print(f"Error: {e}")  # More detailed error logging
        raise ValueError("Error processing Gemini response")


def run_bulk_scraper(urls: List[str]):
    results = []

    def scrape_url(url):
        try:
            html = fetch_html_selenium(url)
            data = extract_data(html, url)
            return {"url": url, "data": data}
        except Exception as e:
            return {"url": url, "error": str(e)}

    for url in urls:
        results.append(scrape_url(url))

    return {"results": results}
