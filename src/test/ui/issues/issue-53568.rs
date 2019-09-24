// Regression test for an NLL-related ICE (#53568) -- we failed to
// resolve inference variables in "custom type-ops".
//
// build-pass (FIXME(62277): could be check-pass?)

trait Future {
    type Item;
}

impl<F, T> Future for F
where F: Fn() -> T
{
    type Item = T;
}

trait Connect {}

struct Connector<H> {
    handler: H,
}

impl<H, T> Connect for Connector<H>
where
    T: 'static,
    H: Future<Item = T>
{
}

struct Client<C> {
    connector: C,
}

fn build<C>(_connector: C) -> Client<C> {
    unimplemented!()
}

fn client<H>(handler: H) -> Client<impl Connect>
where H: Fn() + Copy
{
    let connector = Connector {
        handler,
    };
    let client = build(connector);
    client
}

fn main() { }
