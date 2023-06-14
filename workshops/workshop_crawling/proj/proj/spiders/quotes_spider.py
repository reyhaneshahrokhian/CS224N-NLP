import json
import jmespath
from pathlib import Path
from collections import defaultdict
import pandas as pd
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://songsara.net/117184/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'quotes-{page}.html'
        Path(filename).write_bytes(response.body)
        self.log(f'Saved file {filename}')
        self.parse_post(response)

    def parse_post(self, response):
        """
        Parses a post page
        """
        if len(response.css(
                '.dl-box')) > 1:  # our spider does not support such pages with multiple music tracks like https://tinyurl.com/2p85b4hx
            return []

        schema_graph_data = json.loads(response.xpath('//script[@type="application/ld+json"]//text()').get())
        info = jmespath.search('"@graph"[?"@type"==\'Article\']|[0]', schema_graph_data)
        dictionary = defaultdict(lambda: list())
        if info:
            songs = response.xpath('//ul[has-class("audioplayer-audios")]//li')
            artists = response.xpath('//div[has-class("AR-Si")]//a')
            for song in songs:
                music_name = song.attrib.get('data-title', '')
                album_name = song.attrib.get('data-album', '')
                artist_name = song.attrib.get('data-artist', '')
                file_urls = song.attrib.get('data-src', '')
                dictionary['music_name'].append(music_name)
                dictionary['album_name'].append(album_name)
                dictionary['artist_name'].append(artist_name)
                dictionary['file_url'].append(file_urls)

            df = pd.DataFrame.from_dict(dictionary)
            df.to_csv('./data.csv')