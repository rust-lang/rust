// This regression test exposed an issue where ZSTs were not being placed at the
// beinning of generator field layouts, causing an assertion error downstream.

// compile-pass
// edition:2018

#![feature(async_await)]
#![allow(unused)]

async fn foo<F>(_: &(), _: F) {}

fn main() {
    foo(&(), || {});
    async {
        foo(&(), || {}).await;
    };
}
