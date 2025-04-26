#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn test<const N: usize>() -> [u8; N - 1] {
    //~^ ERROR overflow
    todo!()
}

fn main() {
    test::<0>();
}
