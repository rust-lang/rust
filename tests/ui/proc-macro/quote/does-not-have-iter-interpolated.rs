#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::quote;

fn main() {
    let nonrep = "";

    // Without some protection against repetitions with no iterator somewhere
    // inside, this would loop infinitely.
    quote!($($nonrep)*); //~ ERROR mismatched types
}
