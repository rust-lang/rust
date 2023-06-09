#![allow(dead_code)]

struct Foo { a: u8 }
fn bar() -> Foo {
    Foo { a: 5 }
}

static foo: Foo = bar();
//~^ ERROR cannot call non-const fn

fn main() {}
