// build-fail
// aux-build:main_functions.rs

#![feature(imported_main)]

extern crate main_functions;
pub use main_functions::boilerplate as main; //~ ERROR entry symbol `main` from foreign crate

// FIXME: Should be run-pass
