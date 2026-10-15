#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) -> [u8; N] {
    [0; N]
}

trait Tr {
    fn m(const N: usize);
    fn n(x: usize);
}

struct S;

impl Tr for S {
    fn m(x: usize) {} //~ ERROR E0049
    fn n(const N: usize) {} //~ ERROR E0049
}

fn main() {
    let _ = foo::<3>(3); //~ ERROR E0107
    let _ = foo(true); //~ ERROR E0308
    //~^ ERROR
}
