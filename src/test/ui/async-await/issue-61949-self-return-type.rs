// edition:2018

// This test checks that `Self` is prohibited as a return type. See #61949 for context.

pub struct Foo<'a> {
    pub bar: &'a i32,
}

impl<'a> Foo<'a> {
    pub async fn new(_bar: &'a i32) -> Self {
    //~^ ERROR `async fn` return type cannot contain a projection or `Self` that references lifetimes from a parent scope
        Foo {
            bar: &22
        }
    }
}

async fn foo() {
    let x = {
        let bar = 22;
        Foo::new(&bar).await
    };
    drop(x);
}

fn main() { }
