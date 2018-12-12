// edition:2018

#![feature(uniform_paths)]

// Built-in attribute
use inline as imported_inline;

#[imported_inline] //~ ERROR cannot use a built-in attribute through an import
fn main() {}
