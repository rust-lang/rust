#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn add<const N: usize>() -> [u8; { N + 1; 5 }] {
    //~^ ERROR overly complex generic constant
    todo!()
}

fn div<const N: usize>() -> [u8; { N / 1; 5 }] {
    //~^ ERROR overly complex generic constant
    todo!()
}

const fn foo(n: usize) {}

fn fn_call<const N: usize>() -> [u8; { foo(N); 5 }] {
    //~^ ERROR overly complex generic constant
    todo!()
}

fn main() {
    add::<12>();
    div::<9>();
    fn_call::<14>();
}
