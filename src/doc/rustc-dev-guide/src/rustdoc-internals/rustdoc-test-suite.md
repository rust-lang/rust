# The `rustdoc` test suite

This page is about the test suite named `rustdoc` used to test the HTML output of `rustdoc`.
For other rustdoc-specific test suites, see [Rustdoc test suites].

Each test file in this test suite is simply a Rust source file `file.rs` sprinkled with
so-called *directives* located inside normal Rust code comments.
These come in two flavors: *Compiletest* and *HtmlDocCk*.

To learn more about the former, read [Compiletest directives].
For the latter, continue reading.

Internally, [`compiletest`] invokes the supplementary checker script [`htmldocck.py`].

[Rustdoc test suites]: ../tests/compiletest.md#rustdoc-test-suites
[`compiletest`]: ../tests/compiletest.md
[`htmldocck.py`]: https://github.com/rust-lang/rust/blob/master/src/etc/htmldocck.py

## HtmlDocCk Directives

Directives to HtmlDocCk are assertions that place constraints on the generated HTML.
They look similar to those given to `compiletest` in that they take the form of `//@` comments
but ultimately, they are completely distinct and processed by different programs.

[XPath] is used to query parts of the HTML document tree.

**Introductory example**:

```rust,ignore (illustrative)
//@ has file/type.Alias.html
//@ has - '//*[@class="rust item-decl"]//code' 'type Alias = Option<i32>;'
pub type Alias = Option<i32>;
```

Here, we check that documentation generated for crate `file` contains a page for the
public type alias `Alias` where the code block that is found at the top contains the
expected rendering of the item. The `//*[@class="rust item-decl"]//code` is an XPath
expression.

Conventionally, you place these directives directly above the thing they are meant to test.
Technically speaking however, they don't need to be as HtmlDocCk only looks for the directives.

All directives take a `PATH` argument.
To avoid repetition, `-` can be passed to it to re-use the previous `PATH` argument.
Since the path contains the name of the crate, it is conventional to add a
`#![crate_name = "foo"]` attribute to the crate root to shorten the resulting path.

All arguments take the form of shell-style (single or double) quoted strings,
with the exception of `COUNT` and the special `-` form of `PATH`.

All directives (except `files`) can be *negated* by putting a `!` in front of their name.
Before you add negated directives, please read about [their caveats](#caveats).

Similar to shell commands,
directives can extend across multiple lines if their last char is `\`.
In this case, the start of the next line should be `//`, with no `@`.

Similar to compiletest directives, besides a space you can also use a colon `:` to separate
the directive name and the arguments, however a space is preferred for HtmlDocCk directives.

Use the special string `{{channel}}` in XPaths, `PATTERN` arguments and [snapshot files](#snapshot)
if you'd like to refer to the URL `https://doc.rust-lang.org/CHANNEL` where `CHANNEL` refers to the
current release channel (e.g, `stable` or `nightly`).

Listed below are all possible directives:

[XPath]: https://en.wikipedia.org/wiki/XPath

### `has`

> Usage 1: `//@ has PATH`

Check that the file given by `PATH` exists.

> Usage 2: `//@ has PATH XPATH PATTERN`

Checks that the text of each element / attribute / text selected by `XPATH` in the
whitespace-normalized[^1] file given by `PATH` matches the
(also whitespace-normalized) string `PATTERN`.

**Tip**: If you'd like to avoid whitespace normalization and/or if you'd like to match with a regex,
use `matches` instead.

### `hasraw`

> Usage: `//@ hasraw PATH PATTERN`

Checks that the contents of the whitespace-normalized[^1] file given by `PATH`
matches the (also whitespace-normalized) string `PATTERN`.

**Tip**: If you'd like to avoid whitespace normalization and / or if you'd like to match with a
regex, use `matchesraw` instead.

### `matches`

> Usage: `//@ matches PATH XPATH PATTERN`

Checks that the text of each element / attribute / text selected by `XPATH` in the
file given by `PATH` matches the Python-flavored[^2] regex `PATTERN`.

### `matchesraw`

> Usage: `//@ matchesraw PATH PATTERN`

Checks that the contents of the file given by `PATH` matches the
Python-flavored[^2] regex `PATTERN`.

### `count`

> Usage: `//@ count PATH XPATH COUNT`

Checks that there are exactly `COUNT` matches for `XPATH` within the file given by `PATH`.

### `snapshot`

> Usage: `//@ snapshot NAME PATH XPATH`

Checks that the element / text selected by `XPATH` in the file given by `PATH` matches the
pre-recorded subtree or text (the "snapshot") in file `FILE_STEM.NAME.html` where `FILE_STEM`
is the file stem of the test file.

Pass the `--bless` option to `compiletest` to accept the current subtree/text as expected.
This will overwrite the aforementioned file (or create it if it doesn't exist). It will
automatically normalize the channel-dependent URL `https://doc.rust-lang.org/CHANNEL` to
the special string `{{channel}}`.

### `has-dir`

> Usage: `//@ has-dir PATH`

Checks for the existence of the directory given by `PATH`.

### `files`

> Usage: `//@ files PATH ENTRIES`

Checks that the directory given by `PATH` contains exactly `ENTRIES`.
`ENTRIES` is a Python-like list of strings inside a quoted string.

**Example**: `//@ files "foo/bar" '["index.html", "sidebar-items.js"]'`

[^1]: Whitespace normalization means that all spans of consecutive whitespace are replaced with a single space.
[^2]: They are Unicode aware (flag `UNICODE` is set), match case-sensitively and in single-line mode.

## Compiletest Directives (Brief)

As mentioned in the introduction, you also have access to [compiletest directives].
Most importantly, they allow you to register auxiliary crates and
to pass flags to the `rustdoc` binary under test.
It's *strongly recommended* to read that chapter if you don't know anything about them yet.

Here are some details that are relevant to this test suite specifically:

* While you can use both `//@ compile-flags` and `//@ doc-flags` to pass flags to `rustdoc`,
  prefer to user the latter to show intent. The former is meant for `rustc`.
* Add `//@ build-aux-docs` to the test file that has auxiliary crates to not only compile the
  auxiliaries with `rustc` but to also document them with `rustdoc`.

## Caveats

Testing for the absence of an element or a piece of text is quite fragile and not very future proof.

It's not unusual that the *shape* of the generated HTML document tree changes from time to time.
This includes for example renamings of CSS classes.

Whenever that happens, *positive* checks will either continue to match the intended element /
attribute / text (if their XPath expression is general / loose enough) and
thus continue to test the correct thing or they won't in which case they would fail thereby
forcing the author of the change to look at them.

Compare that to *negative* checks (e.g., `//@ !has PATH XPATH PATTERN`) which won't fail if their
XPath expression "no longer" matches. The author who changed "the shape" thus won't get notified and
as a result someone else can unintentionally reintroduce `PATTERN` into the generated docs without
the original negative check failing.

**Note**: Please avoid the use of *negated* checks!

**Tip**: If you can't avoid it, please **always** pair it with an analogous positive check in the
immediate vicinity, so people changing "the shape" have a chance to notice and to update the
negated check!

## Limitations

HtmlDocCk uses the XPath implementation from the Python standard library.
This leads to several limitations:

* All `XPATH` arguments must start with `//` due to a flaw in the implementation.
* Many XPath features (functions, axies, etc.) are not supported.
* Only well-formed HTML can be parsed (hopefully rustdoc doesn't output mismatched tags).

Furthmore, compiletest [revisions] are not supported.

[revisions]: ../tests/compiletest.md#revisions
[compiletest directives]: ../tests/directives.md
