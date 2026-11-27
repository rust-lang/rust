//@ edition:2021

trait SendFuture: Send {
    type Output;
}

impl<Fut: Send> SendFuture for Fut {
    type Output = ();
}

async fn broken_fut() {
    ident_error;
    //~^ ERROR cannot find value `ident_error` in this scope
}

// triggers normalization of `<Fut as SendFuture>::Output`,
// which requires `Fut: Send`.
fn normalize<Fut: SendFuture>(_: Fut, _: Fut::Output) {}

async fn iceice<A, B>()
// <- async fn is necessary
where
    A: Send,
    B: Send, // <- a second bound
{
    normalize(broken_fut(), ());
}

fn main() {}
