//@ edition: 2021

use std::future::Future;

async fn bop() {
    fold(run(), |mut foo| async move {
        &mut foo.bar;
    })
}

fn fold<Fut, F, U>(_: Foo<U>, f: F)
where
    F: FnMut(Foo<U>) -> Fut,
{
    loop {}
}

struct Foo<F> {
    bar: Vec<F>,
}

fn run() -> Foo<impl Future<Output = ()>> {
    //~^ ERROR type annotations needed
    loop {}
}

fn main() {}
