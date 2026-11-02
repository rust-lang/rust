#![feature(gca_adts, gca_min_const_items, gca_macroless_args, min_adt_const_params)]

pub fn takes_nested_tuple<const N: u32>() {
    takes_nested_tuple::<{ () }> //~ ERROR expected `u32`, found `()`
}

fn main() {}
