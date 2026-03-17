#![feature(generic_const_exprs)]
#![allow(incomplete_features, todo_macro_uses)]

fn test<const N: usize>() -> [u8; N - 1] {
    //~^ ERROR overflow
    todo!()
}

fn main() {
    test::<0>();
}
