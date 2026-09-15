// https://github.com/rust-lang/rust/issues/95873
#![crate_name = "foo"]

//@ has foo/index.html "//dt" "pub use ::std as x;"
pub use ::std as x;
