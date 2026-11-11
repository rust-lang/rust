#![feature(type_alias_impl_trait)]

// this test used to stack overflow due to infinite recursion.
//@ check-pass
//@ edition: 2018

use std::future::Future;

fn main() {
    let _ = move || async move {
        let value = 0u8;
        blah(&value).await;
    };
}

type BlahFut<'a> = impl Future<Output = ()> + Send + 'a;
#[define_opaque(BlahFut)]
fn blah<'a>(_value: &'a u8) -> BlahFut<'a> {
    async {}
}
