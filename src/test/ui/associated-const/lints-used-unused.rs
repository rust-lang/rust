// Tests that associated constants are checked whether they are used or not.
//
// revisions: used unused
// compile-flags: -Copt-level=2 --emit link

#![cfg_attr(unused, allow(dead_code))]
#![deny(arithmetic_overflow)]

pub trait Foo {
    const N: i32;
}

struct S;

impl Foo for S {
    const N: i32 = 1 << 42;
    //~^ ERROR this arithmetic operation will overflow
}

impl<T: Foo> Foo for Vec<T> {
    const N: i32 = --T::N + (-i32::MIN); //~ ERROR this arithmetic operation will overflow
}

fn main() {
    #[cfg(used)]
    let _ = S::N; //[used]~ ERROR erroneous constant used
}
