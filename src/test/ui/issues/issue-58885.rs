// compile-pass
// edition:2018

#![feature(async_await, await_macro, futures_api)]

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth<'a>(
        &'a self, foo: &'a dyn Foo
    ) -> bool
    {
        true
    }
}

fn main() {}
