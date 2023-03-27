// Regression test for <https://github.com/rust-lang/rust/issues/59368>.
// The goal is to ensure that `doc(hidden)`, `doc(inline)` and `doc(no_inline)`
// are not copied from an item when inlined.

#![crate_name = "foo"]
#![feature(doc_cfg)]
#![feature(no_core)]
#![no_core]

// @has 'foo/index.html'

#[doc(hidden, cfg(feature = "foo"))]
pub struct Foo;

#[doc(hidden, no_inline, cfg(feature = "bar"))]
pub use Foo as Foo1;

#[doc(hidden, inline)]
pub use Foo1 as Foo2;

// First we ensure that only the reexport `Bar2` and the inlined struct `Bar`
// are inlined.
// @count - '//a[@class="struct"]' 2
// Then we check that both `cfg` are displayed.
// @matches - '//*[@class="stab portability"]' '^foo$'
// And finally we check that the only element displayed is `Bar`.
// @has - '//a[@href="struct.Bar.html"]' 'Bar'
#[doc(inline)]
pub use Foo as Bar;

// Re-exported `#[doc(hidden)]` items are inlined as well.
// @has - '//a[@href="struct.Bar2.html"]' 'Bar2'
// @matches - '//*[@class="stab portability"]' '^bar and foo$'
pub use Foo2 as Bar2;
