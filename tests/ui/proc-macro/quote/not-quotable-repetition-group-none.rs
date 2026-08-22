#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::quote;

macro_rules! do_quote {
    ($dollar:tt $content:expr) => {
        proc_macro::quote!($dollar $content *) //~ ERROR the trait bound `[{integer}; 3]: ToTokens` is not satisfied [E0277]
    };
}

fn main() {
    let arr = [1, 2, 3];
    do_quote!($ f!($arr));
}
