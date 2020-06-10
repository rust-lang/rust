#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

pub fn arr_len<const N: usize>() {
    let _: [u8; N + 1];
    //~^ ERROR constant expression depends on a generic parameter
}

struct Const<const N: usize>;

pub fn func_call<const N: usize>() {
    let _: Const::<{N + 1}>;
    //~^ ERROR constant expression depends on a generic parameter
}

fn main() {}
