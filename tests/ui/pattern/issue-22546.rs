//@ run-pass
#![allow(unused_variables)]
// Parsing patterns with paths with type parameters (issue #22544)

use std::default::Default;

#[derive(Default)]
pub struct Foo<T>(T, T);

impl<T: std::fmt::Display> Foo<T> {
    fn foo(&self) {
        match *self {
            Foo::<T>(ref x, ref y) => println!("Goodbye, World! {} {}", x, y)
        }
    }
}

trait Tr { //~ WARN trait `Tr` is never used
    type U;
}

impl<T> Tr for Foo<T> {
    type U = T;
}

struct Wrapper<T> {
    value: T
}

fn main() {
    let Foo::<i32>(a, b) = Default::default();

    let f = Foo(2,3);
    f.foo();

    let w = Wrapper { value: Foo(10u8, 11u8) };
    match w {
        Wrapper::<Foo<u8>> { value: Foo(10, 11) } => {},
        crate::Wrapper::<<Foo<_> as Tr>::U> { value: Foo::<u8>(11, 16) } => { panic!() },
        _ => { panic!() }
    }

    if let None::<u8> = Some(8) {
        panic!();
    }
    if let None::<u8> { .. } = Some(8) {
        panic!();
    }
    if let Option::None::<u8> { .. } = Some(8) {
        panic!();
    }
}
