//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn make_array<const M: usize>() -> [(); M + 1] {
    [(); M + 1]
}

fn foo<const N: usize>() -> [(); (N * 2) + 1] {
    make_array::<{ N * 2 }>()
}

fn main() {
    assert_eq!(foo::<10>(), [(); 10 * 2 + 1])
}

// Tests that N * 2 is considered const_evalutable by appearing as part of the (N * 2) + 1 const
