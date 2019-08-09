// edition:2015

#![feature(async_await)]

async fn foo() {} //~ ERROR `async fn` is not permitted in the 2015 edition

fn baz() { async fn foo() {} } //~ ERROR `async fn` is not permitted in the 2015 edition

async fn async_baz() { //~ ERROR `async fn` is not permitted in the 2015 edition
    async fn bar() {} //~ ERROR `async fn` is not permitted in the 2015 edition
}

struct Foo {}

impl Foo {
    async fn foo() {} //~ ERROR `async fn` is not permitted in the 2015 edition
}

trait Bar {
    async fn foo() {} //~ ERROR `async fn` is not permitted in the 2015 edition
                      //~^ ERROR trait fns cannot be declared `async`
}

fn main() {
    macro_rules! accept_item { ($x:item) => {} }

    accept_item! {
        async fn foo() {} //~ ERROR `async fn` is not permitted in the 2015 edition
    }

    accept_item! {
        impl Foo {
            async fn bar() {} //~ ERROR `async fn` is not permitted in the 2015 edition
        }
    }

    let inside_closure = || {
        async fn bar() {} //~ ERROR `async fn` is not permitted in the 2015 edition
    };
}
