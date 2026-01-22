//@ aux-build:issue-61592.rs
// https://github.com/rust-lang/rust/issues/61592
#![crate_name="bar"]

extern crate foo;

//@ has bar/index.html
//@ has - '//a[@href="#reexports"]' 'Re-exports'
//@ has - '//code' 'pub use foo::FooTrait as _;'
//@ !has - '//a[@href="trait._.html"]' ''
pub use foo::FooTrait as _;

//@ has bar/index.html
//@ has - '//a[@href="#reexports"]' 'Re-exports'
//@ has - '//code' 'pub use foo::FooStruct as _;'
//@ !has - '//a[@href="struct._.html"]' ''
pub use foo::FooStruct as _;
