//@ aux-build:variant-struct.rs
//@ build-aux-docs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/32395
#![crate_name="issue_32395"]

//@ has variant_struct/enum.Foo.html
//@ !hasraw - 'pub qux'
//@ !hasraw - 'pub(crate) qux'
//@ !hasraw - 'pub Bar'
extern crate variant_struct;

//@ has issue_32395/enum.Foo.html
//@ !hasraw - 'pub qux'
//@ !hasraw - 'pub(crate) qux'
//@ !hasraw - 'pub Bar'
pub use variant_struct::Foo;
