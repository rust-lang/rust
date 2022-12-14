#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];
//~^ ERROR evaluation of `Arr::<0>::{constant#0}` failed

fn test<const N: usize>() -> Arr<N>
where
    [u8; N - 1]: Sized,
    //~^ ERROR evaluation of `test::<0>::{constant#0}` failed
{
    todo!()
}

fn main() {
    test::<0>();
}
