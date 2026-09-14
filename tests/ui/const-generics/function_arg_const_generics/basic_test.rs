//@ check-pass
#![feature(function_arg_const_generics, min_generic_const_args)]

fn foo(const N: usize) -> [u8; N] { [0; N] }

fn forward<const K: usize>() -> [u8; K] { foo(K) }

struct S;

impl S {
    fn m(&self, x: u8, const N: usize) -> [u8; N] { [x; N] }
}

fn main() {
    let _: [u8; 3] = foo(3);
    let _: [u8; 5] = forward::<5>();
    let _: [u8; 4] = S::m(&S, 1, 4);
}
