// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_evaluatable_checked)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];
//[min]~^ ERROR generic parameters may not be used in const operations

fn test<const N: usize>() -> Arr<N> where Arr<N>: Sized {
    todo!()
}

fn main() {
    test::<0>();
    //[full]~^ ERROR failed to evaluate the given constant
    //[full]~| ERROR failed to evaluate the given constant
}
