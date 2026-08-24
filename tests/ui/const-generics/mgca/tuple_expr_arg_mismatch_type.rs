#![feature(min_generic_const_args, macroless_generic_const_args)]
#![expect(incomplete_features)]

pub fn takes_nested_tuple<const N: u32>() {
    takes_nested_tuple::<{ () }> //~ ERROR expected `u32`, found `()`
}

fn main() {}
