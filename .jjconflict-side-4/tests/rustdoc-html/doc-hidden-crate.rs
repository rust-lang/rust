// Regression test for <https://github.com/rust-lang/rust/issues/126796>.
// `doc(hidden)` should still be able to hide extern crates, only the local crates
// cannot be hidden because we still need to generate its `index.html` file.

#![crate_name = "foo"]
#![doc(hidden)]

//@ has 'foo/index.html'
// First we check that the page contains the crate name (`foo`).
//@ has - '//*' 'foo'
// But doesn't contain any of the other items.
//@ !has - '//*' 'other'
//@ !has - '//*' 'marker'
//@ !has - '//*' 'PhantomData'

#[doc(inline)]
pub use std as other;

#[doc(inline)]
pub use std::marker;

#[doc(inline)]
pub use std::marker::PhantomData;

//@ !has - '//*' 'myself'
#[doc(inline)]
pub use crate as myself;
