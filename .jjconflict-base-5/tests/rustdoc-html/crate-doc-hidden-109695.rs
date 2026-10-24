// This test ensures that even if the crate module is `#[doc(hidden)]`, the file
// is generated.
// https://github.com/rust-lang/rust/issues/109695

//@ has 'foo/index.html'
//@ has 'foo/all.html'

#![crate_name = "foo"]
#![doc(hidden)]
