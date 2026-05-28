//@ check-pass
//@ edition:2021

#![allow(dead_code)]

struct Foo<'a>(&'a u32);

impl<'a> Foo<'a> {
    async fn foo() {
        struct Bar<'b>(&'b u32);

        impl<'b> Bar<'b> {
            async fn bar() {}
        }
    }
}

fn main() {}
