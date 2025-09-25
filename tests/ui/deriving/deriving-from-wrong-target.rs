//@ check-fail

#![feature(derive_from)]
#![allow(dead_code)]

use std::from::From;

#[derive(From)]
//~^ ERROR `#[derive(From)]` used on a struct with no fields
struct S1;

#[derive(From)]
//~^ ERROR `#[derive(From)]` used on a struct with no fields
struct S2 {}

#[derive(From)]
//~^ ERROR `#[derive(From)]` used on a struct with multiple fields
struct S3(u32, bool);

#[derive(From)]
//~^ ERROR `#[derive(From)]` used on a struct with multiple fields
struct S4 {
    a: u32,
    b: bool,
}

#[derive(From)]
//~^ ERROR `#[derive(From)]` used on an enum
enum E1 {}

#[derive(From)]
struct SUnsizedField<T: ?Sized> {
    last: T,
    //~^ ERROR the size for values of type `T` cannot be known at compilation time [E0277]
}

fn main() {}
