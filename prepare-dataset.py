import requests
from bs4 import BeautifulSoup
import urllib.parse
import os


def fetch_html(url):
    """Fetch HTML content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None
    

def parse_paper_links(html_content, base_url):
    """Extract the paper mentioned in the page."""
    soup = BeautifulSoup(html_content, 'html.parser')
    papers_url = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'poster' in href:
            paper_url = urllib.parse.urljoin(base_url, href)
            papers_url.append(paper_url)

    return papers_url


def parse_poster_link(html):
    """Extract the poster image link from the page."""
    soup = BeautifulSoup(html, "html.parser")
    poster_link = None
    for link in soup.find_all("a", href=True):
        if "PosterPDFs" in link["href"] and '.png' in link["href"]:
            poster_link = urllib.parse.urljoin("https://icml.cc", link["href"])  # Ensure full URL
            break
    return poster_link

def download_poster(url, save_dir):
    """Download and save the poster image."""
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(save_dir, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_dir}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def main():
    base_url = 'https://icml.cc/virtual/2023/events/poster'
    save_dir = '../icml_2023_posters'
    os.makedirs(save_dir, exist_ok=True)

    html_content = fetch_html(base_url)
    paper_links = parse_paper_links(html_content, base_url)

    print('Number of papers:', len(paper_links))
    for link in paper_links:
        html = fetch_html(link)
        if not html:
            continue

        poster_link = parse_poster_link(html)
        if poster_link:
            filename = f"{link.split('/')[-1]}.png"
            save_path = os.path.join(save_dir, filename)

            if not os.path.exists(save_path):
                download_poster(poster_link, save_path)

if __name__ == '__main__':
    main()
