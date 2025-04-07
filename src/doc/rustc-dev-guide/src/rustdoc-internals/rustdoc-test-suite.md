# The `rustdoc` test suite

This page is specifically about the test suite named `rustdoc`.
For other test suites used for testing rustdoc, see [Rustdoc tests](../rustdoc.md#tests).

The `rustdoc` test suite is specifically used to test the HTML output of rustdoc.

This is achieved by means of `htmldocck.py`, a custom checker script that leverages [XPath].

[XPath]: https://en.wikipedia.org/wiki/XPath

## Directives
Directives to htmldocck are similar to those given to `compiletest` in that they take the form of `//@` comments.

In addition to the directives listed here,
`rustdoc` tests also support most
[compiletest directives](../tests/directives.html).

All `PATH`s in directives are relative to the the rustdoc output directory (`build/TARGET/test/rustdoc/TESTNAME`),
so it is conventional to use a `#![crate_name = "foo"]` attribute to avoid
having to write a long crate name multiple times.
To avoid repetion, `-` can be used in any `PATH` argument to re-use the previous `PATH` argument.

All arguments take the form of quoted strings
(both single and double quotes are supported),
with the exception of `COUNT` and the special `-` form of `PATH`.

Directives are assertions that place constraints on the generated HTML.

All directives (except `files`) can be negated by putting a `!` in front of their name.

Similar to shell commands,
directives can extend across multiple lines if their last char is `\`.
In this case, the start of the next line should be `//`, with no `@`.

For example, `//@ !has 'foo/struct.Bar.html'` checks that crate `foo` does not have a page for a struct named `Bar` in the crate root.

### `has`

Usage 1: `//@ has PATH`
Usage 2: `//@ has PATH XPATH PATTERN`

In the first form, `has` checks that a given file exists.

In the second form, `has` is an alias for `matches`,
except `PATTERN` is a whitespace-normalized[^1] string instead of a regex.

### `matches`

Usage: `//@ matches PATH XPATH PATTERN`

Checks that the text of each element selected by `XPATH` in `PATH` matches the python-flavored regex `PATTERN`.

### `matchesraw`

Usage: `//@ matchesraw PATH PATTERN`

Checks that the contents of the file `PATH` matches the regex `PATTERN`.

### `hasraw`

Usage: `//@ hasraw PATH PATTERN`

Same as `matchesraw`, except `PATTERN` is a whitespace-normalized[^1] string instead of a regex.

### `count`

Usage: `//@ count PATH XPATH COUNT`

Checks that there are exactly `COUNT` matches for `XPATH` within the file `PATH`.

### `snapshot`

Usage: `//@ snapshot NAME PATH XPATH`

Creates a snapshot test named NAME.
A snapshot test captures a subtree of the DOM, at the location
determined by the XPath, and compares it to a pre-recorded value
in a file. The file's name is the test's name with the `.rs` extension
replaced with `.NAME.html`, where NAME is the snapshot's name.

htmldocck supports the `--bless` option to accept the current subtree
as expected, saving it to the file determined by the snapshot's name.
compiletest's `--bless` flag is forwarded to htmldocck.

### `has-dir`

Usage: `//@ has-dir PATH`

Checks for the existance of directory `PATH`.

### `files`

Usage: `//@ files PATH ENTRIES`

Checks that the directory `PATH` contains exactly `ENTRIES`.
`ENTRIES` is a python list of strings inside a quoted string,
as if it were to be parsed by `eval`.
(note that the list is actually parsed by `shlex.split`,
so it cannot contain arbitrary python expressions).

Example: `//@ files "foo/bar" '["index.html", "sidebar-items.js"]'`

[^1]: Whitespace normalization means that all spans of consecutive whitespace are replaced with a single space.  The files themselves are also whitespace-normalized.

## Limitations
`htmldocck.py` uses the xpath implementation from the standard library.
This leads to several limitations:
* All `XPATH` arguments must start with `//` due to a flaw in the implemention.
* Many XPath features (functions, axies, etc.) are not supported.
* Only well-formed HTML can be parsed (hopefully rustdoc doesn't output mismatched tags).

