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

/// This tests units. See also #835.
/// kiB MiB GiB TiB PiB EiB
/// kib Mib Gib Tib Pib Eib
/// kB MB GB TB PB EB
/// kb Mb Gb Tb Pb Eb
/// 32kiB 32MiB 32GiB 32TiB 32PiB 32EiB
/// 32kib 32Mib 32Gib 32Tib 32Pib 32Eib
/// 32kB 32MB 32GB 32TB 32PB 32EB
/// 32kb 32Mb 32Gb 32Tb 32Pb 32Eb
fn test_units() {
}

/// This test has [a link_with_underscores][chunked-example] inside it. See #823.
/// See also [the issue tracker](https://github.com/Manishearth/rust-clippy/search?q=doc_markdown&type=Issues). And here is another [inline link][inline_link].
///
/// [chunked-example]: https://en.wikipedia.org/wiki/Chunked_transfer_encoding#Example
/// [inline_link]: https://foobar

/// The `main` function is the entry point of the program. Here it only calls the `foo_bar` and
/// `multiline_ticks` functions.
///
/// expression of the type  `_ <bit_op> m <cmp_op> c` (where `<bit_op>`
/// is one of {`&`, '|'} and `<cmp_op>` is one of {`!=`, `>=`, `>` ,
fn main() {
//~^ ERROR: you should put `link_with_underscores` between ticks
    foo_bar();
    multiline_ticks();
    test_emphasis();
    test_units();
}
