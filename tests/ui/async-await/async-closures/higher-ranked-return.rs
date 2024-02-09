// aux-build:block-on.rs
// edition:2021

// known-bug: unknown
// Borrow checking doesn't like that higher-ranked output...

#![feature(async_closure)]

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let x = async move |x: &str| -> &str {
            x
        };
        let s = x("hello!").await;
    });
}
