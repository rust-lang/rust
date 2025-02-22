#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Bug<T, U> = impl Fn(T) -> U + Copy;
//~^ ERROR cycle detected

const CONST_BUG: Bug<u8, ()> = unsafe { std::mem::transmute(|_: u8| ()) };
//~^ ERROR item does not constrain `Bug::{opaque#0}`, but has it in its signature
//~| ERROR item does not constrain `Bug::{opaque#0}`, but has it in its signature
//~| ERROR non-defining opaque type use in defining scope

fn make_bug<T, U: From<T>>() -> Bug<T, U> {
    |x| x.into()
}

fn main() {
    CONST_BUG(0);
}
