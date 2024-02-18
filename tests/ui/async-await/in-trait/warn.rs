//@ edition: 2021

#![deny(async_fn_in_trait)]

pub trait Foo {
    async fn not_send();
    //~^ ERROR  use of `async fn` in public traits is discouraged
}

mod private {
    pub trait FooUnreachable {
        async fn not_send();
        // No warning
    }
}

pub(crate) trait FooCrate {
    async fn not_send();
    // No warning
}

fn main() {}
