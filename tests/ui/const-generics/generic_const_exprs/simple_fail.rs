#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

type Arr<const N: usize> = [u8; N - 1];
//~^ ERROR overflow

fn test<const N: usize>() -> Arr<N>
where
    [u8; N - 1]: Sized,
    //~^ ERROR overflow
{
    todo!()
}

fn main() {
    test::<0>();
}
