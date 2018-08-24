// aux-build:cross-glob.rs
// build-aux-docs
// ignore-cross-compile

extern crate inner;

// @has cross_glob/struct.SomeStruct.html
// @has cross_glob/fn.some_fn.html
// @!has cross_glob/index.html '//code' 'pub use inner::*;'
#[doc(inline)]
pub use inner::*;
