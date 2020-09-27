#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

fn test<const N: usize>() -> [u8; N - 1] {
    //~^ ERROR evaluation of constant
    todo!()
}

fn main() {
    test::<0>();
}
