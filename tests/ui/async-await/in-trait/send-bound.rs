// check-pass
// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait MyTrait {
    #[async_send]
    async fn foo(&self) -> usize;
}

fn assert_send<T: Send>(_: T) {}

fn use_trait<T: MyTrait>(x: T) {
    assert_send(x.foo())
}

fn main() {}
