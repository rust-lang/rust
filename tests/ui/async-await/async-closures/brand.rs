// aux-build:block-on.rs
// edition:2021
// build-pass

#![feature(async_closure, async_fn_traits)]

extern crate block_on;

use std::future::Future;
use std::marker::PhantomData;
use std::ops::AsyncFn;

struct S;
struct B<'b>(PhantomData<&'b mut &'b mut ()>);

impl S {
    async fn q<F: AsyncFn(B<'_>)>(self, f: F) {
        f(B(PhantomData)).await;
    }
}

fn main() {
    block_on::block_on(async {
        S.q(async |b: B<'_>| { println!("...") }).await;
    });
}
