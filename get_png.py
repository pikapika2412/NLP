import requests
from bs4 import BeautifulSoup
import os
import time
import urllib.parse

# Base URL structure
BASE_URL = "https://icml.cc/virtual/2023/poster/"
POSTER_IDS = [24493]
SAVE_DIR = "icml_2023_posters"

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_html(url):
    """Fetch HTML content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def extract_poster_link(html):
    """Extract the poster image link from the page."""
    soup = BeautifulSoup(html, "html.parser")
    poster_link = None
    for link in soup.find_all("a", href=True):
        if "PosterPDFs" in link["href"] and '.png' in link["href"]:
            poster_link = urllib.parse.urljoin("https://icml.cc", link["href"])  # Ensure full URL
            break
    return poster_link

def download_poster(url, save_path):
    """Download and save the poster image."""
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def main():
    for poster_id in POSTER_IDS:
        poster_url = f"{BASE_URL}{poster_id}"
        print(f"Fetching: {poster_url}")

        html = fetch_html(poster_url)
        if not html:
            continue

        poster_link = extract_poster_link(html)
        if poster_link:
            filename = f"poster_{poster_id}.png"
            save_path = os.path.join(SAVE_DIR, filename)

            if not os.path.exists(save_path):
                download_poster(poster_link, save_path)
            else:
                print(f"Already downloaded: {save_path}")
        else:
            print(f"No poster found for: {poster_url}")

        time.sleep(1)  # Be polite, avoid overloading the server

if __name__ == "__main__":
    main()
