#![feature(type_info)]
//@ run-pass
//@ check-run-results
#![allow(dead_code)]

use std::mem::type_info::Type;

struct Foo {
    a: u32,
}

enum Bar {
    Some(u32),
    None,
    Foomp { a: (), b: &'static str },
}

struct Unsized {
    x: u16,
    s: str,
}

fn main() {
    println!("{:#?}", const { Type::of::<(u8, u8, ())>() }.kind);
    println!("{:#?}", const { Type::of::<[u8; 2]>() }.kind);
    println!("{:#?}", const { Type::of::<Foo>() }.kind);
    println!("{:#?}", const { Type::of::<Bar>() }.kind);
    println!("{:#?}", const { Type::of::<&Unsized>() }.kind);
    println!("{:#?}", const { Type::of::<&str>() }.kind);
    println!("{:#?}", const { Type::of::<&[u8]>() }.kind);
    println!("{:#?}", const { Type::of::<str>() }.kind);
    println!("{:#?}", const { Type::of::<[u8]>() }.kind);
}
