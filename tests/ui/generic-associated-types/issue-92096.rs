// edition:2018

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
    //~^ ERROR `C` does not live long enough
    //
    // FIXME(#71723). This is because we infer at some point a value of
    //
    // impl Future<Output = <C as Client>::Connection<'_>>
    //
    // and then we somehow fail the WF check because `where C: 'a` is not known,
    // but I'm not entirely sure how that comes about.
}

fn main() {}
