//@ edition:2018

// This test checks that we emit the correct borrowck error when `Self` is used as a return type.
// See #61949 for context.

pub struct Foo<'a> {
    pub bar: &'a i32,
}

impl<'a> Foo<'a> {
    pub async fn new(_bar: &'a i32) -> Self {
        Foo {
            bar: &22
        }
    }
}

pub async fn foo() {
    let x = {
        let bar = 22;
        Foo::new(&bar).await
        //~^ ERROR `bar` does not live long enough
    };
    drop(x);
}

fn main() { }
