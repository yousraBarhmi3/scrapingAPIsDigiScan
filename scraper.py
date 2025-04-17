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

    service = Service(executable_path="/usr/local/bin/chromedriver")

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


def parse_response(content):
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and all(
            isinstance(item, dict) and "type" in item and "url" in item
            for item in parsed
        ):
            return parsed
        else:
            raise ValueError("Parsed content is not a list of {type, url} objects.")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Response content: {content}")
        raise ValueError("Model response is not valid JSON.")


def extract_links(links: List[str], selected_model: str) -> List[Dict[str, str]]:
    user_prompt = f"{USER_MESSAGE} {json.dumps(links, ensure_ascii=False)}"

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
        
        print("Raw response from the model:", content)  # Print the raw response
        return parse_response(content)

    elif selected_model == "gemini-1.5-flash":
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "url": {"type": "string"}
                        },
                        "required": ["type", "url"]
                    }
                }
            }
        )
        response = model.generate_content(f"{LINKS_MESSAGE}\n{USER_MESSAGE} {json.dumps(links)}")
        content = response.candidates[0].content.parts[0].text
        
        print("Raw response from the model:", content)  # Print the raw response
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
            
        print("Raw response from the model:", content)  # Print the raw response
        return parse_response(content)

    else:
        raise ValueError(f"Unsupported model: {selected_model}")

MAX_TEXT_BLOCKS = 15
def extract_data(html_content, url_info):
    soup = BeautifulSoup(html_content, "lxml")
    url_type = url_info.get("type", "").lower()
    page_url = url_info.get("url")

    # === 🔹 Meta Basics ===
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    description_tag = soup.find("meta", attrs={"name": "description"})
    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    description = description_tag.get("content", "").strip() if description_tag else ""
    keywords = keywords_tag.get("content", "").strip() if keywords_tag else ""
    html_lang = soup.html.get("lang", "").strip() if soup.html and soup.html.has_attr("lang") else ""

    # === 🔹 Headings & Content ===
    headings = {
        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
        "h4": [h.get_text(strip=True) for h in soup.find_all("h4")],
        "h5": [h.get_text(strip=True) for h in soup.find_all("h5")],
        "h6": [h.get_text(strip=True) for h in soup.find_all("h6")]
    }
    text_elements = soup.find_all(["p", "li"])
    text_blocks = [t.get_text(strip=True) for t in text_elements if t.get_text(strip=True)]
    word_count = len(" ".join(text_blocks).split())

    # === 🔹 Links
    base_domain = urlparse(page_url).netloc
    all_links = [urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True)]
    internal_links = {link for link in all_links if urlparse(link).netloc == base_domain}
    external_links = {link for link in all_links if urlparse(link).netloc != base_domain}

    # === 🔹 Media / Visuals
    images = [{"src": img.get("src"), "alt": img.get("alt", "").strip()} for img in soup.find_all("img")]

    # === 🔹 CTA Detection
    links_buttons = []
    for tag in soup.find_all(["a", "button"]):
        text = tag.get_text(strip=True).lower()
        if text:
            links_buttons.append({
                "text": text,
                "href": tag.get("href")
            })

    # === 🔹 Social Links
    social_platforms = ["facebook", "linkedin", "instagram", "youtube", "tiktok", "x.com", "twitter"]
    social_links = [
        a.get("href") for a in soup.find_all("a", href=True)
        if any(platform in a.get("href", "") for platform in social_platforms)
    ]

    # === 🔹 Tracking Scripts Detection
    scripts = soup.find_all("script", src=True)
    tracking_patterns = ["analytics", "gtag", "gtm.js", "matomo", "facebook.net", "hotjar"]
    has_tracking_scripts = any(
        any(tp in script["src"] for tp in tracking_patterns)
        for script in scripts
    )

    # === 🧩 Contact Page Specifics
    contact_info = {}
    thank_you_url = None
    
    # Keywords related to the thank-you page URL
    thank_you_keywords = [
        "thank-you", "thanks", "success", "confirmation", "received", 
        "merci", "reussi", "confirmation", "reception", "remerciement", "reçu"
    ]

    if url_type == "contact":
        forms = soup.find_all("form")
        has_form = bool(forms)
        has_email = bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", html_content))
        has_phone = bool(re.search(r"\+?\d[\d\-\s\(\)]{6,}", html_content))
        has_map = bool(soup.find("iframe", src=re.compile("google.com/maps")))
        has_rgpd = bool(re.search(r"(rgpd|politique de confidentialité|données personnelles)", html_content, re.I))
        
        # Check thank-you URL in form action
        for form in forms:
            action = form.get("action", "")
            if any(kw in action.lower() for kw in thank_you_keywords):
                thank_you_url = urljoin(page_url, action)
                break

        # Check thank-you redirection via window.location in scripts
        scripts_with_redirection = soup.find_all("script", string=True)
        for script in scripts_with_redirection:
            script_content = script.string or ""
            # Match lines like: window.location = '/thank-you.html';
            matches = re.findall(r"window\.location(?:\.href)?\s*=\s*['\"]([^'\"]+)['\"]", script_content, flags=re.IGNORECASE)
            for potential_url in matches:
                if any(kw in potential_url.lower() for kw in thank_you_keywords):
                    thank_you_url = urljoin(page_url, potential_url)
                    break
            if thank_you_url:
                break
        contact_info.update({
            "has_form": has_form,
            "has_email": has_email,
            "has_phone": has_phone,
            "has_map": has_map,
            "has_rgpd": has_rgpd,
            "thank_you_url": thank_you_url
        })

    # === ✅ Final structured output
    data = {
        "url": page_url,
        "type": url_type,
        "title": title,
        "meta_description": description,
        "meta_keywords": keywords,
        "lang": html_lang,
        "headings": headings,
        "text_blocks": text_blocks[:MAX_TEXT_BLOCKS],
        "word_count": word_count,
        "internal_links": len(internal_links),
        "external_links": len(external_links),
        "links & buttons": links_buttons,
        "images": images,
        "social_links": social_links,
        "has_tracking_scripts": has_tracking_scripts,
    }

    # Merge contact-specific info
    if contact_info:
        data.update(contact_info)

    return data

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


def run_bulk_scraper(typed_links: List[Dict[str, str]]):
    results = []

    def scrape_url(link_info):
        try:
            html = fetch_html_selenium(link_info["url"])
            data = extract_data(html, link_info)  # on passe bien l'objet entier {type, url}
            return {"url": link_info["url"], "data": data}
        except Exception as e:
            return {"url": link_info["url"], "error": str(e)}

    for link in typed_links:
        results.append(scrape_url(link))

    return {"results": results}


def extract_linkedIn_ads_data(html_data):
    soup = BeautifulSoup(html_data, "html.parser")
    ads = []
    
    # Each ad is inside a <li> with class "search-result-item"
    ad_items = soup.find_all("li", class_="search-result-item")
    
    for ad in ad_items:
        ad_data = {}
        
        # Extract title (Company name)
        title_tag = ad.find("div", class_="text-md")
        ad_data["title"] = title_tag.get_text(strip=True) if title_tag else None

        # Extract description
        desc_tag = ad.find("p", class_="commentary__content")
        ad_data["description"] = desc_tag.get_text(strip=True) if desc_tag else None

        # Extract ad link (first <a> with href that starts with /ad-library/detail/)
        link_tag = ad.find("a", href=lambda x: x and x.startswith("/ad-library/detail/"))
        if link_tag:
            ad_data["url"] = "https://www.linkedin.com" + link_tag["href"]
        else:
            ad_data["url"] = None
        
        ads.append(ad_data)
    
    return ads
   
def extract_linkedIn_ad_detail(html_data: str) -> dict:
    soup = BeautifulSoup(html_data, "html.parser")
    
    data = {
        "advertiser": None,
        "description": None,
        "headline": None,
        "image_url": None,
        "external_link": None,
        "ad_type": None,
        "duration": None,
        "paying_entity": None, 
        "impressions": {},
        "targeting": {
            "language": [],
            "location_includes": [],
            "location_excludes": [],
            "audience": None
        }
    }

    # Advertiser
    adv_tag = soup.select_one("a[aria-label^='View organization page']")
    if adv_tag:
        data["advertiser"] = adv_tag.get_text(strip=True)

    # Description
    desc_tag = soup.select_one("p.commentary__content")
    if desc_tag:
        data["description"] = desc_tag.get_text(strip=True)

    # Headline
    headline_tag = soup.select_one("h2.text-sm.font-semibold")
    if headline_tag:
        data["headline"] = headline_tag.get_text(strip=True)

    # Ad Image
    image_tag = soup.select_one("img.ad-preview__dynamic-dimensions-image")
    if image_tag and image_tag.has_attr("src"):
        data["image_url"] = image_tag["src"]

    # External Link
    link_tag = soup.select_one("a[href^='http']")
    if link_tag and link_tag.has_attr("href"):
        data["external_link"] = link_tag["href"]

    # Ad Type
    ad_type_tag = soup.select_one("p.text-sm.mb-1")
    if ad_type_tag:
        data["ad_type"] = ad_type_tag.get_text(strip=True)

    # Duration
    duration_tag = soup.select_one("p.about-ad__availability-duration")
    if duration_tag:
        data["duration"] = duration_tag.get_text(strip=True)

    # Paying Entity
    paying_entity_tag = soup.select_one("p.about-ad__paying-entity")
    if paying_entity_tag:
        data["paying_entity"] = paying_entity_tag.get_text(strip=True)

    # Country-wise impressions
    impressions = soup.select("span.ad-analytics__country-impressions")
    for imp in impressions:
        country = imp.select_one("p.font-semibold")
        percent = imp.select_one("p.text-right")
        if country and percent:
            data["impressions"][country.get_text(strip=True)] = percent.get_text(strip=True)

    # Targeting - Language
    lang_tag = soup.select_one("h3:contains('Language') + p span")
    if lang_tag:
        data["targeting"]["language"].append(lang_tag.get_text(strip=True))

    # Targeting - Locations
    location_block = soup.select("span.ad-targeting__segments")
    for block in location_block:
        text = block.get_text()
        if "includes" in text:
            data["targeting"]["location_includes"].append(text.replace("Targeting includes ", "").strip())
        elif "excludes" in text:
            data["targeting"]["location_excludes"].append(text.replace("Targeting excludes ", "").strip())

    # Audience Targeting
    audience = soup.select_one("h3:contains('Audience') + p")
    if audience:
        data["targeting"]["audience"] = audience.get_text(strip=True)

    return data
