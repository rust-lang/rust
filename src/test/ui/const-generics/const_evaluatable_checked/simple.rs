// [full] run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]
#![feature(const_evaluatable_checked)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];
//[min]~^ ERROR generic parameters must not be used inside of non trivial constant values

fn test<const N: usize>() -> Arr<N> where Arr<N>: Default {
    Default::default()
}

fn main() {
    let x = test::<33>();
    assert_eq!(x, [0; 32]);
}
