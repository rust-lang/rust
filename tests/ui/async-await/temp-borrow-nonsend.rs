//@ check-pass
//@ edition:2021

use core::marker::PhantomData;

struct B(PhantomData<*const ()>);

fn do_sth(_: &B) {}

async fn foo() {}

async fn test() {
    let b = B(PhantomData);
    do_sth(&b);
    drop(b);
    foo().await;
}

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(test());
}
