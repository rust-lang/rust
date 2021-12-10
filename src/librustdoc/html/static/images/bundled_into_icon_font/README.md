The SVGs in this directory are bundled into a webfont. This allows them to be easily colored with CSS.

## Creating the webfont

Use https://fontello.com

1. Drag all SVG images in this folder onto the webpage to upload them
2. After upload is complete, select them all by dragging a rectangle over them under "**Custom Icons**"
3. Go to the "Customize Codes" tab
4. We use the following for each icon

| Icon           | Unicode Character | Unicode Code |
|--------------- |-------------------|--------------|
| ![](caret-down.svg)           | ⏷  | 23F7         |
| ![](clipboard.svg)            | 📋 | 1F4CB        |
| ![](cog.svg)                  | ⚙  | 2699         |
| ![](exclamation-triangle.svg) | ⚠  | 26A0         |
| ![](flask.svg)                | 🔬 | 1F52C        |
| ![](info-circle.svg)          | ⓘ | 24D8         |
| ![](paint-brush.svg)          | 🖌 | 1F58C        |
| ![](thumbs-down.svg)          | 👎 | 1F44E        |
| ![](toggle-minus.svg)         | ⊟  | 229F         |
| ![](toggle-plus.svg)          | ⊞  | 229E         |

We set these characters to fitting Unicode symbols for the fallback case if a user's browser doesn't support our webfonts or has webfonts disabled.

5. Click the wrench icon, then go to "Advanced font settings"
6. Select Unicode encoding
7. Set "font name" to `icons`
8. Download webfont
9. Extract `font/icons.woff` and `font/icons.woff2` into `static/fonts`
