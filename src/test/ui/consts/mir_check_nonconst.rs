#![allow(dead_code)]

struct Foo { a: u8 }
fn bar() -> Foo {
    Foo { a: 5 }
}

static foo: Foo = bar();
//~^ ERROR calls in statics are limited to constant functions, tuple structs and tuple variants

fn main() {}
