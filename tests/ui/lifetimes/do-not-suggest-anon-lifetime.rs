//@ aux-build:anon_lifetime.rs
//@ edition:2024
extern crate anon_lifetime;
use anon_lifetime::{Service, Context};

struct Foo;

impl<S, Request: 'static> Service<S, Request> for Foo {
    type Response = ();
    type Error = ();

    fn serve(
        &self,
        _ctx: Context<S>,
        _req: Request,
    ) -> impl Future<Output = Result<Self::Response, Self::Error>> {
        //~^ ERROR: the parameter type `S` may not live long enough
        async { todo!() }
    }
}

fn main() {}
