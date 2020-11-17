// [full] run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_evaluatable_checked)]
#![allow(incomplete_features)]

fn test<const N: usize>() -> [u8; N - 1] where [u8; N - 1]: Default {
    //[min]~^ ERROR generic parameters
    //[min]~| ERROR generic parameters
    Default::default()
}

fn main() {
    let x = test::<33>();
    assert_eq!(x, [0; 32]);
}
