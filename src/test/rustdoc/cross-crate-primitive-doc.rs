// aux-build:primitive-doc.rs
// compile-flags: --extern-html-root-url=primitive_doc=../ -Z unstable-options

#![no_std]

extern crate primitive_doc;

// @has 'cross_crate_primitive_doc/fn.foo.html' '//a[@href="../primitive_doc/primitive.usize.html"]' 'usize'
pub fn foo() -> usize { 0 }
