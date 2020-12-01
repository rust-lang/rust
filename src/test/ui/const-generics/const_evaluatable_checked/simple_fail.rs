// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]
#![feature(const_evaluatable_checked)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1]; //[full]~ ERROR evaluation of constant
//[min]~^ ERROR generic parameters may not be used in const operations

fn test<const N: usize>() -> Arr<N> where Arr<N>: Sized {
    todo!()
}

fn main() {
    test::<0>();
}
