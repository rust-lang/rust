//@ aux-build:bad_main_functions.rs

extern crate bad_main_functions;
pub use bad_main_functions::boilerplate as main;

//~? ERROR `main` function has wrong type
