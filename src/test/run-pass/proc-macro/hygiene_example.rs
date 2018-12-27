#![allow(unused_macros)]
// aux-build:hygiene_example_codegen.rs
// aux-build:hygiene_example.rs

#![feature(proc_macro_hygiene)]

extern crate hygiene_example;
use hygiene_example::hello;

fn main() {
    mod hygiene_example {} // no conflict with `extern crate hygiene_example;` from the proc macro
    macro_rules! format { () => {} } // does not interfere with `format!` from the proc macro
    macro_rules! hello_helper { () => {} } // similarly does not intefere with the proc macro

    let string = "world"; // no conflict with `string` from the proc macro
    hello!(string);
    hello!(string);
}
