// aux-build:rustdoc-extern-default-method.rs
// ignore-cross-compile

extern crate rustdoc_extern_default_method as ext;

// @count extern_default_method/struct.Struct.html '//*[@id="method.provided"]' 1
pub use ext::Struct;
