// issue - https://github.com/rust-lang/rust/issues/161101
//@ build-pass
//@ compile-flags: --crate-type=lib
//@ edition: 2024
#![feature(async_drop)]
#![feature(gen_blocks)]
#![allow(incomplete_features)]

pub async fn html() {
    async gen {
        gen { yield };
        yield
    };
}
