// compile-pass
// aux-build:bang_proc_macro.rs

#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate bang_proc_macro;

fn main() {
    bang_proc_macro!(println!("Hello, world!"));
}
