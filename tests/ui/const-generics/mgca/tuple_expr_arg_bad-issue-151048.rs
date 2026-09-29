#![feature(gca_min_const_items, gca_macroless_args)]
#![expect(incomplete_features)]

struct Y {
    stuff: [u8; { ([1, 2], 3, [4, 5]) }], //~ ERROR expected `usize`, found `([1, 2], 3, [4, 5])`
}

fn main() {}
