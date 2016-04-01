//! This file tests for the DOC_MARKDOWN lint
//~^ ERROR: you should put `DOC_MARKDOWN` between ticks

#![feature(plugin)]
#![plugin(clippy)]

#![deny(doc_markdown)]

/// The foo_bar function does _nothing_. See also foo::bar. (note the dot there)
/// Markdown is _weird_. I mean _really weird_.  This \_ is ok. So is `_`. But not Foo::some_fun
/// which should be reported only once despite being __doubly bad__.
fn foo_bar() {
//~^ ERROR: you should put `foo_bar` between ticks
//~| ERROR: you should put `foo::bar` between ticks
//~| ERROR: you should put `Foo::some_fun` between ticks
}

/// That one tests multiline ticks.
/// ```rust
/// foo_bar FOO_BAR
/// _foo bar_
/// ```
fn multiline_ticks() {
}

/// This _is a test for
/// multiline
/// emphasis_.
fn test_emphasis() {
}

/// This test has [a link with underscores][chunked-example] inside it. See #823.
/// See also [the issue tracker](https://github.com/Manishearth/rust-clippy/search?q=doc_markdown&type=Issues).
///
/// [chunked-example]: http://en.wikipedia.org/wiki/Chunked_transfer_encoding#Example

/// The `main` function is the entry point of the program. Here it only calls the `foo_bar` and
/// `multiline_ticks` functions.
fn main() {
    foo_bar();
    multiline_ticks();
    test_emphasis();
}
