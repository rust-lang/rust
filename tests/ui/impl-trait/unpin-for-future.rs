//@ edition:2024
//
// Tests that you can't implement Unpin for a compiler-generated future using TAIT.

#![feature(type_alias_impl_trait)]

use core::marker::PhantomPinned;
use core::pin::Pin;

type MyFut = impl Future<Output = ()>;

async fn my_async_fn() {}

#[define_opaque(MyFut)]
fn fut() -> MyFut {
    my_async_fn()
}

impl Unpin for MyFut {}
//~^ ERROR: only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
