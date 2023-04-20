//@aux-build:../auxiliary/proc_macros.rs:proc-macro

extern crate proc_macros;

use proc_macros::with_span;

fn main() {
    println!(with_span!(""something ""));
}
