//@ run-pass
//@ aux-build:main_functions.rs

#![feature(imported_main)]

extern crate main_functions;
pub use main_functions::boilerplate as main;
