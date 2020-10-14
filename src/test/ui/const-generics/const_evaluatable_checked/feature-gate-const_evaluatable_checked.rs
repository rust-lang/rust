// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

type Arr<const N: usize> = [u8; N - 1];
//[min]~^ ERROR generic parameters may not be used in const operations

fn test<const N: usize>() -> Arr<N> where Arr<N>: Default {
    //[full]~^ ERROR constant expression depends
    Default::default()
}

fn main() {
    let x = test::<33>();
    assert_eq!(x, [0; 32]);
}
