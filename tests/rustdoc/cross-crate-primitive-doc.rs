// aux-build:primitive-doc.rs
// compile-flags: --extern-html-root-url=primitive_doc=../ -Z unstable-options
// only-linux

#![feature(no_core)]
#![no_core]

extern crate primitive_doc;

// @has 'cross_crate_primitive_doc/fn.foo.html' '//a[@href="../primitive_doc/primitive.usize.html"]' 'usize'
// @has 'cross_crate_primitive_doc/fn.foo.html' '//a[@href="../primitive_doc/primitive.usize.html"]' 'link'
/// [link](usize)
pub fn foo() -> usize { 0 }
