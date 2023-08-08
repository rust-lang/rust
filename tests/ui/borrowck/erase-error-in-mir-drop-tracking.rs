// compile-flags: -Zdrop-tracking-mir
// edition:2021

use std::future::Future;

trait Client {
    type Connecting<'a>: Future + Send
    where
        Self: 'a;

    fn connect(&'_ self) -> Self::Connecting<'a>;
    //~^ ERROR use of undeclared lifetime name `'a`
}

fn call_connect<C>(c: &'_ C) -> impl '_ + Future + Send
where
    C: Client + Send + Sync,
{
    async move { c.connect().await }
    //~^ ERROR `C` does not live long enough
}

fn main() {}
