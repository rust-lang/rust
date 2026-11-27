//@ run-rustfix
#![allow(dead_code)]
struct Foo {
    field: i32,
}

impl Foo {
    fn foo<'a>(&self, x: &'a i32) -> &i32 {
        x
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
