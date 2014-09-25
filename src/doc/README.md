# Rust documentations

## Dependencies

[Pandoc](http://johnmacfarlane.net/pandoc/installing.html), a universal
document converter, is required to generate docs as HTML from Rust's
source code.

[po4a](http://po4a.alioth.debian.org/) is required for generating translated
docs from the master (English) docs.

[GNU gettext](http://www.gnu.org/software/gettext/) is required for managing
the translation data.

## Building

To generate all the docs, just run `make docs` from the root of the repository.
This will convert the distributed Markdown docs to HTML and generate HTML doc
for the 'std' and 'extra' libraries.

To generate HTML documentation from one source file/crate, do something like:

~~~~
rustdoc --output html-doc/ --output-format html ../src/libstd/path.rs
~~~~

(This, of course, requires a working build of the `rustdoc` tool.)

## Additional notes

To generate an HTML version of a doc from Markdown manually, you can do
something like:

~~~~
pandoc --from=markdown --to=html5 --number-sections -o reference.html reference.md
~~~~

(`reference.md` being the Rust Reference Manual.)

The syntax for pandoc flavored markdown can be found at:

- http://johnmacfarlane.net/pandoc/README.html#pandocs-markdown

A nice quick reference (for non-pandoc markdown) is at:

- http://kramdown.gettalong.org/quickref.html

## Notes for translators

Notice: The procedure described below is a work in progress. We are working on
translation system but the procedure contains some manual operations for now.

To start the translation for a new language, see `po4a.conf` at first.

To generate `.pot` and `.po` files, do something like:

~~~~
po4a --copyright-holder="The Rust Project Developers" \
    --package-name="Rust" \
    --package-version="0.13.0" \
    -M UTF-8 -L UTF-8 \
    src/doc/po4a.conf
~~~~

(the version number must be changed if it is not `0.13.0` now.)

Now you can translate documents with `.po` files, commonly used with gettext. If
you are not familiar with gettext-based translation, please read the online
manual linked from http://www.gnu.org/software/gettext/ . We use UTF-8 as the
file encoding of `.po` files.

When you want to make a commit, do the command below before staging your
change:

~~~~
for f in src/doc/po/**/*.po; do
    msgattrib --translated $f -o $f.strip
    if [ -e $f.strip ]; then
       mv $f.strip $f
    else
       rm $f
    fi
done
~~~~

This removes untranslated entries from `.po` files to save disk space.
