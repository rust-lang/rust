// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, feature(const_evaluatable_checked))]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1]; //[full]~ ERROR evaluation of constant
//[min]~^ ERROR generic parameters may not be used in const operations

fn test<const N: usize>() -> Arr<N> where [u8; N - 1]: Sized {
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR evaluation of constant
    todo!()
}

fn main() {
    test::<0>();
}
