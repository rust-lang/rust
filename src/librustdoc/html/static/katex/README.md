# [<img src="https://khan.github.io/KaTeX/katex-logo.svg" width="130" alt="KaTeX">](http://khan.github.io/KaTeX/) [![Build Status](https://travis-ci.org/Khan/KaTeX.svg?branch=master)](https://travis-ci.org/Khan/KaTeX)

KaTeX is a fast, easy-to-use JavaScript library for TeX math rendering on the web.

 * **Fast:** KaTeX renders its math synchronously and doesn't need to reflow the page. See how it compares to a competitor in [this speed test](http://jsperf.com/katex-vs-mathjax/).
 * **Print quality:** KaTeX’s layout is based on Donald Knuth’s TeX, the gold standard for math typesetting.
 * **Self contained:** KaTeX has no dependencies and can easily be bundled with your website resources.
 * **Server side rendering:** KaTeX produces the same output regardless of browser or environment, so you can pre-render expressions using Node.js and send them as plain HTML.

KaTeX supports all major browsers, including Chrome, Safari, Firefox, Opera, and IE 8 - IE 11.

## Usage

Download the built files from [the releases page](https://github.com/khan/katex/releases). Include the `katex.min.js` and `katex.min.css` files on your page:

```html
<link rel="stylesheet" type="text/css" href="/path/to/katex.min.css">
<script src="/path/to/katex.min.js" type="text/javascript"></script>
```

Call `katex.render` with a TeX expression and a DOM element to render into:

```js
katex.render("c = \\pm\\sqrt{a^2 + b^2}", element);
```

To generate HTML on the server, you can use `katex.renderToString`:

```js
var html = katex.renderToString("c = \\pm\\sqrt{a^2 + b^2}");
// '<span class="katex">...</span>'
```

Make sure to include the CSS and font files, but there is no need to include the JavaScript.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

KaTeX is licenced under the [MIT License](http://opensource.org/licenses/MIT).
