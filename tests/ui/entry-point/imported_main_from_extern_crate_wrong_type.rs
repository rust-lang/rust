// aux-build:bad_main_functions.rs

#![feature(imported_main)]

extern crate bad_main_functions;
pub use bad_main_functions::boilerplate as main;
