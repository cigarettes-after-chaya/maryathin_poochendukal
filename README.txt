BLOOM — PYTHON-RENDERED FLOWER GARDEN
====================================

WHAT'S INCLUDED
- index.html: website; selection screen appears first, no flower animation plays until chosen.
- media/lily.mp4
- media/sunflower.mp4
- media/rose.mp4
- media/poppy.mp4
- media/passiflora.mp4
- bloom_renderer.py: procedural Python renderer used to create the MP4s.

OPEN LOCALLY
1. Extract the ZIP, keeping the media folder beside index.html.
2. Open index.html in a modern browser. If your phone's file manager cannot play local MP4s,
   publish the folder to GitHub Pages or use a small local web server.

PUBLISH TO GITHUB PAGES
1. Create/open a GitHub repository.
2. Upload index.html, bloom_renderer.py, README.txt, and the entire media folder.
3. In repository Settings > Pages, select the branch/folder containing index.html.
4. Open the Pages URL. The website uses relative media paths, so the MP4s must stay in media/.

RENDER THE VIDEOS AGAIN
Requires Python 3, NumPy, OpenCV-Python, and FFmpeg installed on PATH.

    pip install numpy opencv-python
    python bloom_renderer.py

Render one flower only:
    python bloom_renderer.py --flower lily
    python bloom_renderer.py --flower sunflower
    python bloom_renderer.py --flower rose
    python bloom_renderer.py --flower poppy
    python bloom_renderer.py --flower passiflora

The renderer creates vertical 540x960, 15 fps MP4 animations. It uses procedural parametric
petal surfaces, staged growth, shading, leaves, botanical centers, and floating particles.
It is a custom lightweight Python/OpenCV renderer inspired by the supplied 3D lily script;
it is not a full physically based 3D engine like Blender or the original high-density renderer.
