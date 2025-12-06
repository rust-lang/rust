#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::quote;

fn main() {
    let arr = [1, 2, 3];
    quote! { $($arr) $$ x .. false [] * }; //~ ERROR the trait bound `[{integer}; 3]: ToTokens` is not satisfied [E0277]
}
