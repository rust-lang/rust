// aux-build:bang_proc_macro.rs

#![feature(proc_macro_non_items)]

#[macro_use]
extern crate bang_proc_macro;

fn main() {
    bang_proc_macro!(println!("Hello, world!"));
    //~^ ERROR: procedural macros cannot be imported with `#[macro_use]`
}
