#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Bug<T, U> = impl Fn(T) -> U + Copy;

union Moo {
    x: Bug<u8, ()>,
    y: (),
}

const CONST_BUG: Bug<u8, ()> = unsafe { Moo { y: () }.x };
//~^ ERROR non-defining opaque type use

fn make_bug<T, U: From<T>>() -> Bug<T, U> {
    |x| x.into() //~ ERROR the trait bound `U: From<T>` is not satisfied
}

fn main() {
    CONST_BUG(0);
}
