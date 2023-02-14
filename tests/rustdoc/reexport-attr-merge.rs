// Regression test for <https://github.com/rust-lang/rust/issues/59368>.
// The goal is to ensure that `doc(hidden)`, `doc(inline)` and `doc(no_inline`)

#![crate_name = "foo"]
#![feature(doc_cfg)]

// @has 'foo/index.html'

#[doc(hidden, cfg(feature = "foo"))]
pub struct Foo;

#[doc(hidden, no_inline, cfg(feature = "bar"))]
pub use Foo as Foo1;

#[doc(hidden, inline)]
pub use Foo1 as Foo2;

// First we ensure that none of the other items are generated.
// @count - '//a[@class="struct"]' 1
// Then we check that both `cfg` are displayed.
// @has - '//*[@class="stab portability"]' 'foo'
// @has - '//*[@class="stab portability"]' 'bar'
// And finally we check that the only element displayed is `Bar`.
// @has - '//a[@class="struct"]' 'Bar'
#[doc(inline)]
pub use Foo2 as Bar;
