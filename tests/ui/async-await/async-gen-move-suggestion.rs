// This is a regression test for <https://github.com/rust-lang/rust/issues/139839>.
// It ensures that the "add `move` keyword" suggestion is valid.

//@ run-rustfix
//@ edition:2024

#![feature(coroutines)]
#![feature(gen_blocks)]
#![feature(async_iterator)]

use std::async_iter::AsyncIterator;

#[allow(dead_code)]
fn moved() -> impl AsyncIterator<Item = u32> {
    let mut x = "foo".to_string();

    async gen { //~ ERROR
        x.clear();
        for x in 3..6 { yield x }
    }
}

#[allow(dead_code)]
fn check_with_whitespace_chars() -> impl AsyncIterator<Item = u32> {
    let mut x = "foo".to_string();

    async // Just to check that whitespace characters are correctly handled
    gen { //~^ ERROR
        x.clear();
        for x in 3..6 { yield x }
    }
}

fn main() {
}
