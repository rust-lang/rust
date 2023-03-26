// run-rustfix
//! This file tests for the `DOC_MARKDOWN` lint.

#![allow(dead_code, incomplete_features)]
#![warn(clippy::doc_markdown)]
#![feature(custom_inner_attributes, generic_const_exprs, const_option)]
#![rustfmt::skip]

/// The foo_bar function does _nothing_. See also foo::bar. (note the dot there)
/// Markdown is _weird_. I mean _really weird_. This \_ is ok. So is `_`. But not Foo::some_fun
/// which should be reported only once despite being __doubly bad__.
/// Here be ::a::global:path, and _::another::global::path_.  :: is not a path though.
/// Import an item from ::awesome::global::blob:: (Intended postfix)
/// These are the options for ::Cat: (Intended trailing single colon, shouldn't be linted)
/// That's not code ~NotInCodeBlock~.
/// be_sure_we_got_to_the_end_of_it
fn foo_bar() {
}

/// That one tests multiline ticks.
/// ```rust
/// foo_bar FOO_BAR
/// _foo bar_
/// ```
///
/// ~~~rust
/// foo_bar FOO_BAR
/// _foo bar_
/// ~~~
/// be_sure_we_got_to_the_end_of_it
fn multiline_codeblock() {
}

/// This _is a test for
/// multiline
/// emphasis_.
/// be_sure_we_got_to_the_end_of_it
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
/// NaN
/// be_sure_we_got_to_the_end_of_it
fn test_units() {
}

/// This tests allowed identifiers.
/// KiB MiB GiB TiB PiB EiB
/// DirectX
/// ECMAScript
/// GPLv2 GPLv3
/// GitHub GitLab
/// IPv4 IPv6
/// ClojureScript CoffeeScript JavaScript PureScript TypeScript
/// NaN NaNs
/// OAuth GraphQL
/// OCaml
/// OpenGL OpenMP OpenSSH OpenSSL OpenStreetMap OpenDNS
/// WebGL
/// TensorFlow
/// TrueType
/// iOS macOS FreeBSD
/// TeX LaTeX BibTeX BibLaTeX
/// MinGW
/// CamelCase (see also #2395)
/// be_sure_we_got_to_the_end_of_it
fn test_allowed() {
}

/// This test has [a `link_with_underscores`][chunked-example] inside it. See #823.
/// See also [the issue tracker](https://github.com/rust-lang/rust-clippy/search?q=clippy::doc_markdown&type=Issues)
/// on GitHub (which is a camel-cased word, but is OK). And here is another [inline link][inline_link].
/// It can also be [inline_link2]. A link to [StackOverflow](https://stackoverflow.com) is also acceptable.
///
/// [chunked-example]: https://en.wikipedia.org/wiki/Chunked_transfer_encoding#Example
/// [inline_link]: https://foobar
/// [inline_link2]: https://foobar
/// The `main` function is the entry point of the program. Here it only calls the `foo_bar` and
/// `multiline_ticks` functions.
///
/// expression of the type  `_ <bit_op> m <cmp_op> c` (where `<bit_op>`
/// is one of {`&`, '|'} and `<cmp_op>` is one of {`!=`, `>=`, `>` ,
/// be_sure_we_got_to_the_end_of_it
fn main() {
    foo_bar();
    multiline_codeblock();
    test_emphasis();
    test_units();
}

/// ## CamelCaseThing
/// Talks about `CamelCaseThing`. Titles should be ignored; see issue #897.
///
/// # CamelCaseThing
///
/// Not a title #897 CamelCaseThing
/// be_sure_we_got_to_the_end_of_it
fn issue897() {
}

/// I am confused by brackets? (`x_y`)
/// I am confused by brackets? (foo `x_y`)
/// I am confused by brackets? (`x_y` foo)
/// be_sure_we_got_to_the_end_of_it
fn issue900() {
}

/// Diesel queries also have a similar problem to [Iterator][iterator], where
/// /// More talking
/// returning them from a function requires exposing the implementation of that
/// function. The [`helper_types`][helper_types] module exists to help with this,
/// but you might want to hide the return type or have it conditionally change.
/// Boxing can achieve both.
///
/// [iterator]: https://doc.rust-lang.org/stable/std/iter/trait.Iterator.html
/// [helper_types]: ../helper_types/index.html
/// be_sure_we_got_to_the_end_of_it
fn issue883() {
}

/// `foo_bar
/// baz_quz`
/// [foo
/// bar](https://doc.rust-lang.org/stable/std/iter/trait.IteratorFooBar.html)
fn multiline() {
}

/** E.g., serialization of an empty list: FooBar
```
That's in a code block: `PackedNode`
```

And BarQuz too.
be_sure_we_got_to_the_end_of_it
*/
fn issue1073() {
}

/** E.g., serialization of an empty list: FooBar
```
That's in a code block: PackedNode
```

And BarQuz too.
be_sure_we_got_to_the_end_of_it
*/
fn issue1073_alt() {
}

/// Tests more than three quotes:
/// ````
/// DoNotWarn
/// ```
/// StillDont
/// ````
/// be_sure_we_got_to_the_end_of_it
fn four_quotes() {
}

#[cfg_attr(feature = "a", doc = " ```")]
#[cfg_attr(not(feature = "a"), doc = " ```ignore")]
/// fn main() {
///     let s = "localhost:10000".to_string();
///     println!("{}", s);
/// }
/// ```
fn issue_1469() {}

/**
 * This is a doc comment that should not be a list
 *This would also be an error under a strict common mark interpretation
 */
fn issue_1920() {}

/// An iterator over mycrate::Collection's values.
/// It should not lint a `'static` lifetime in ticks.
fn issue_2210() {}

/// This should not cause the lint to trigger:
/// #REQ-data-family.lint_partof_exists
fn issue_2343() {}

/// This should not cause an ICE:
/// __|_ _|__||_|
fn pulldown_cmark_crash() {}

/// This should not lint
/// (regression test for #7758)
/// [plain text][path::to::item]
fn intra_doc_link() {}

// issue #7033 - generic_const_exprs ICE
struct S<T, const N: usize>
where [(); N.checked_next_power_of_two().unwrap()]: {
    arr: [T; N.checked_next_power_of_two().unwrap()],
    n: usize,
}

impl<T: Copy + Default, const N: usize> S<T, N>
where [(); N.checked_next_power_of_two().unwrap()]: {
    fn new() -> Self {
        Self {
            arr: [T::default(); N.checked_next_power_of_two().unwrap()],
            n: 0,
        }
    }
}
