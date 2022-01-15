// edition:2018
// check-fail
// FIXME(generic_associated_types): this should pass, but we end up
// essentially requiring that `for<'s> C: 's`

#![feature(generic_associated_types)]

use std::future::Future;

trait Client {
    type Connecting<'a>: Future + Send
    where
        Self: 'a;

    fn connect(&'_ self) -> Self::Connecting<'_>;
}

fn call_connect<C>(c: &'_ C) -> impl '_ + Future + Send
//~^ ERROR the parameter
//~| ERROR the parameter
where
    C: Client + Send + Sync,
{
    async move { c.connect().await }
}

fn main() {}
