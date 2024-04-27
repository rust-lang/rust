//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

#![feature(async_closure)]

extern crate block_on;

use std::future::Future;
use std::marker::PhantomData;

struct S;
struct B<'b>(PhantomData<&'b mut &'b mut ()>);

impl S {
    async fn q<F: async Fn(B<'_>)>(self, f: F) {
        f(B(PhantomData)).await;
    }
}

fn main() {
    block_on::block_on(async {
        S.q(async |b: B<'_>| { println!("...") }).await;
    });
}
