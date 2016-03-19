//! This file tests for the DOC_MARKDOWN lint
//~^ ERROR: you should put `DOC_MARKDOWN` between ticks

#![feature(plugin)]
#![plugin(clippy)]

#![deny(doc_markdown)]

/// The foo_bar function does nothing.
//~^ ERROR: you should put `foo_bar` between ticks
fn foo_bar() {
}

/// That one tests multiline ticks.
/// ```rust
/// foo_bar FOO_BAR
/// ```
fn multiline_ticks() {
}

/// The `main` function is the entry point of the program. Here it only calls the `foo_bar` and
/// `multiline_ticks` functions.
fn main() {
    foo_bar();
    multiline_ticks();
}
