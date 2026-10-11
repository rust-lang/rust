#![feature(gca_adts, gca_min_const_items, gca_macroless_args, min_adt_const_params)]

struct Y {
    stuff: [u8; { ([1, 2], 3, [4, 5]) }], //~ ERROR expected `usize`, found `([1, 2], 3, [4, 5])`
}

fn main() {}
