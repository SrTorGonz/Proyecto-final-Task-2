import urllib.request

url = "https://sciviscontest2026.github.io/examples/"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0'}
)
with urllib.request.urlopen(req) as response:
   html = response.read().decode('utf-8')

with open("scratch/examples.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Fetched successfully. Length:", len(html))
