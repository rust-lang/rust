#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Bug<T, U> = impl Fn(T) -> U + Copy;

#[define_opaque(Bug)]
//~^ ERROR: only functions and methods can define opaque types
const CONST_BUG: Bug<u8, ()> = unsafe { std::mem::transmute(|_: u8| ()) };

#[define_opaque(Bug)]
fn make_bug<T, U: From<T>>() -> Bug<T, U> {
    |x| x.into() //~ ERROR is not satisfied
}

fn main() {
    CONST_BUG(0);
}
