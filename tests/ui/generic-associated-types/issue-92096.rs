//@ edition:2018
//@ check-pass

use std::future::Future;

trait Client {
    type Connecting<'a>: Future + Send
    where
        Self: 'a;

    fn connect(&'_ self) -> Self::Connecting<'_>;
}

fn call_connect<C>(c: &'_ C) -> impl '_ + Future + Send
where
    C: Client + Send + Sync,
{
    async move { c.connect().await }
}

fn main() {}
