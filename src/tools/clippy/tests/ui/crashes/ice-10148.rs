//@aux-build:../auxiliary/proc_macros.rs
//@no-rustfix
extern crate proc_macros;

use proc_macros::with_span;

fn main() {
    println!(with_span!(""something ""));
    //~^ println_empty_string
}
