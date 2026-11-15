//! Test interaction between `-Zasync-panic=no` and coroutine types.

//@ build-pass
//@ compile-flags: --crate-type=lib -Zasync-panic=no -Zvalidate-mir
//@ edition: 2024
#![feature(async_drop)]
#![feature(gen_blocks)]
#![allow(incomplete_features)]

pub async fn simple() {
    async gen {
        gen { yield };
        yield
    };
}
