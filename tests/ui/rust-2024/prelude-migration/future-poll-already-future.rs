//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@ check-pass

#![deny(rust_2024_prelude_collisions)]

use std::future::Future;

fn main() {
    core::pin::pin!(async {}).poll(&mut context());
}

fn context() -> core::task::Context<'static> {
    loop {}
}
