# Dependencies

[Pandoc](http://johnmacfarlane.net/pandoc/installing.html), a universal
document converter, is required to generate docs as HTML from Rust's
source code.

[Node.js](http://nodejs.org/) is also required for generating HTML from
the Markdown docs (reference manual, tutorials, etc.) distributed with
this git repository.

# Building

To generate all the docs, just run `make docs` from the root of the repository.
This will convert the distributed Markdown docs to HTML and generate HTML doc
for the 'std' and 'extra' libraries.

To generate HTML documentation from one source file/crate, do something like:

~~~~
rustdoc --output-dir html-doc/ --output-format html ../src/libstd/path.rs
~~~~

(This, of course, requires a working build of the `rustdoc` tool.)

# Additional notes

To generate an HTML version of a doc from Markdown without having Node.js
installed, you can do something like:

~~~~
pandoc --from=markdown --to=html5 --number-sections -o rust.html rust.md
~~~~

(rust.md being the Rust Reference Manual.)

The syntax for pandoc flavored markdown can be found at:
http://johnmacfarlane.net/pandoc/README.html#pandocs-markdown

A nice quick reference (for non-pandoc markdown) is at:
http://kramdown.rubyforge.org/quickref.html
