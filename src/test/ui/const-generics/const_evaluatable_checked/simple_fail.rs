// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, feature(const_evaluatable_checked))]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR evaluation of `Arr::<0_usize>::{constant#0}` failed

fn test<const N: usize>() -> Arr<N> where [u8; N - 1]: Sized {
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR evaluation of `test::<0_usize>::{constant#0}` failed
    todo!()
}

fn main() {
    test::<0>();
}
