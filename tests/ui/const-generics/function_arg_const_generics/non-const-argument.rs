#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) -> [u8; N] {
    [0; N]
}

fn main() {
    let x = 3;
    let _ = foo(x); //~ ERROR
}
