//@ compile-flags: --crate-type=lib
#![allow(incomplete_features)]
#![feature(unnamed_enum_variants)]

#[repr(u8)]
enum Foo {
    _ = 1, //~ ERROR unnamed enum variants are not yet implemented
}
