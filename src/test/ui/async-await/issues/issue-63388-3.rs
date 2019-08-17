// edition:2018
// check-pass

#![feature(async_await)]

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth(
        &self, foo: &dyn Foo
    ) {
    }
}

fn main() {}
