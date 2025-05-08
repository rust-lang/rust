// Regression test for <https://github.com/rust-lang/rust/issues/59368>.
// The goal is to ensure that `doc(hidden)`, `doc(inline)` and `doc(no_inline)`
// are not copied from an item when inlined.

#![crate_name = "foo"]
#![feature(doc_cfg)]

//@ has 'foo/index.html'

#[doc(hidden, cfg(feature = "foo"))]
pub struct Foo;

#[doc(hidden, no_inline, cfg(feature = "bar"))]
pub use Foo as Foo1;

#[doc(hidden, inline)]
pub use Foo1 as Foo2;

// First we ensure that only the reexport `Bar2` and the inlined struct `Bar`
// are inlined.
//@ count - '//a[@class="struct"]' 1
// Then we check that `cfg` is displayed for base item, but not for intermediate re-exports.
//@ has - '//*[@class="stab portability"]' 'foo'
//@ !has - '//*[@class="stab portability"]' 'bar'
// And finally we check that the only element displayed is `Bar`.
//@ has - '//a[@class="struct"]' 'Bar'
#[doc(inline)]
pub use Foo2 as Bar;

// This one should appear but `Bar2` won't be linked because there is no
// `#[doc(inline)]`.
//@ !has - '//*[@id="reexport.Bar2"]' 'pub use Foo2 as Bar2;'
pub use Foo2 as Bar2;
