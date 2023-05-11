// check-pass
// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

pub trait SpiDevice {
    async fn transaction<F, R>(&mut self);
}

impl SpiDevice for () {
    async fn transaction<F, R>(&mut self) {}
}

fn main() {}
