#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1]; //~ ERROR evaluation of constant

fn test<const N: usize>() -> Arr<N> where Arr<N>: Sized {
    todo!()
}

fn main() {
    test::<0>();
}
