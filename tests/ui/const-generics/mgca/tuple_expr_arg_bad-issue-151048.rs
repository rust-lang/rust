#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

struct Y {
    stuff: [u8; { ([1, 2], 3, [4, 5]) }], //~ ERROR expected `usize`, found const tuple
}

fn main() {}
