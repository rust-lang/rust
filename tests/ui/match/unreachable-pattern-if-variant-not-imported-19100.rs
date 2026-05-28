// https://github.com/rust-lang/rust/issues/19100
//@ run-rustfix

#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]

#[derive(Copy, Clone)]
enum Foo {
    Bar,
    Baz
}

impl Foo {
    fn foo(&self) {
        match self {
            &
Bar if true
//~^ ERROR pattern binding `Bar` is named the same as one of the variants of the type `Foo`
=> println!("bar"),
            &
Baz if false
//~^ ERROR pattern binding `Baz` is named the same as one of the variants of the type `Foo`
=> println!("baz"),
_ => ()
        }
    }
}

fn main() {}
