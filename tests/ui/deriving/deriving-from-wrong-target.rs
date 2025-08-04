//@ edition: 2021
//@ check-fail

#![feature(derive_from)]
#![allow(dead_code)]

#[derive(From)]
struct S1;
//~^ ERROR `#[derive(From)]` can only be used on structs with a single field

#[derive(From)]
struct S2 {}
//~^ ERROR `#[derive(From)]` can only be used on structs with a single field

#[derive(From)]
struct S3(u32, bool);
//~^ ERROR `#[derive(From)]` can only be used on structs with a single field

#[derive(From)]
//~v ERROR `#[derive(From)]` can only be used on structs with a single field
struct S4 {
    a: u32,
    b: bool,
}

#[derive(From)]
enum E1 {}
//~^ ERROR `#[derive(From)]` can only be used on structs with a single field

#[derive(From)]
//~v ERROR `#[derive(From)]` can only be used on structs with a single field
enum E2 {
    V1,
    V2,
}

#[derive(From)]
//~v ERROR `#[derive(From)]` can only be used on structs with a single field
enum E3 {
    V1(u32),
    V2(bool),
}

#[derive(From)]
//~^ ERROR the size for values of type `T` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `T` cannot be known at compilation time [E0277]
struct SUnsizedField<T: ?Sized> {
    last: T,
    //~^ ERROR the size for values of type `T` cannot be known at compilation time [E0277]
}

fn main() {}
