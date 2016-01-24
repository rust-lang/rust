#![feature(plugin)]
#![plugin(clippy)]

#![deny(warnings)]
#![allow(dead_code)]

#[derive(PartialEq, Hash)]
struct Foo;

impl PartialEq<u64> for Foo {
    fn eq(&self, _: &u64) -> bool { true }
}

#[derive(Hash)]
//~^ ERROR you are deriving `Hash` but have implemented `PartialEq` explicitly
struct Bar;

impl PartialEq for Bar {
    fn eq(&self, _: &Bar) -> bool { true }
}

#[derive(Hash)]
//~^ ERROR you are deriving `Hash` but have implemented `PartialEq` explicitly
struct Baz;

impl PartialEq<Baz> for Baz {
    fn eq(&self, _: &Baz) -> bool { true }
}

#[derive(Copy)]
struct Qux;

impl Clone for Qux {
//~^ ERROR you are implementing `Clone` explicitly on a `Copy` type
    fn clone(&self) -> Self { Qux }
}

// Ok, `Clone` cannot be derived because of the big array
#[derive(Copy)]
struct BigArray {
    a: [u8; 65],
}

impl Clone for BigArray {
    fn clone(&self) -> Self { unimplemented!() }
}

// Ok, function pointers are not always Clone
#[derive(Copy)]
struct FnPtr {
    a: fn() -> !,
}

impl Clone for FnPtr {
    fn clone(&self) -> Self { unimplemented!() }
}

// Ok, generics
#[derive(Copy)]
struct Generic<T> {
    a: T,
}

impl<T> Clone for Generic<T> {
    fn clone(&self) -> Self { unimplemented!() }
}

fn main() {}
