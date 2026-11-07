//@ edition:2015

async fn foo() {}
//~^ ERROR: `async`/`await` is not permitted in Rust 2015 edition
//~| NOTE: ...and 4 other places
//~| HELP: to use `async`/`await`, switch to Rust 2018
//~| NOTE: for more on editions

fn baz() { async fn foo() {} }

async fn async_baz() {
    async fn bar() {}
}

struct Foo {}

impl Foo {
    async fn foo() {}
}

trait Bar {
    async fn foo() {}
}

fn main() {
    macro_rules! accept_item { ($x:item) => {} }

    accept_item! {
        async fn foo() {}
    }

    accept_item! {
        impl Foo {
            async fn bar() {}
        }
    }

    let inside_closure = || {
        async fn bar() {}
    };
}
