#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

fn test<const N: usize>() -> [u8; N - 1] {
    todo!()
}

fn main() {
    test::<0>();
    //~^ ERROR failed to evaluate the given constant
    //~| ERROR failed to evaluate the given constant
}
