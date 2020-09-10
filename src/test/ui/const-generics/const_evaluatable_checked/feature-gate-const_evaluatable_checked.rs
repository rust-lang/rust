#![feature(const_generics)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];

fn test<const N: usize>() -> Arr<N> where Arr<N>: Default {
    //~^ ERROR constant expression depends
    Default::default()
}

fn main() {
    let x = test::<33>();
    assert_eq!(x, [0; 32]);
}
