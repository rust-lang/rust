//@ aux-build:dep.rs

// test for #121151

#![allow(incomplete_features)]
#![feature(unnamed_fields)]

extern crate dep;

#[repr(C)]
struct A {
    a: u8,
}

enum BadEnum {
    A,
    B,
}

#[repr(C)]
enum BadEnum2 {
    A,
    B,
}

type MyStruct = A;
type MyI32 = i32;

#[repr(C)]
struct L {
    _: i32, //~ ERROR unnamed fields can only have struct or union types
    _: MyI32, //~ ERROR unnamed fields can only have struct or union types
    _: BadEnum, //~ ERROR unnamed fields can only have struct or union types
    _: BadEnum2, //~ ERROR unnamed fields can only have struct or union types
    _: MyStruct,
    _: dep::BadStruct, //~ ERROR named type of unnamed field must have `#[repr(C)]` representation
    _: dep::BadEnum, //~ ERROR unnamed fields can only have struct or union types
    _: dep::BadEnum2, //~ ERROR unnamed fields can only have struct or union types
    _: dep::BadAlias, //~ ERROR unnamed fields can only have struct or union types
    _: dep::GoodAlias,
    _: dep::GoodStruct,
}

fn main() {}
