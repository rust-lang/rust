// aux-build:../../auxiliary/proc_macro_with_span.rs

extern crate proc_macro_with_span;

use proc_macro_with_span::with_span;

fn main() {
    println!(with_span!(""something ""));
}
