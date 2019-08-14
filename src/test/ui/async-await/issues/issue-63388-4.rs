// check-pass
// edition:2018

#![feature(async_await)]

struct A;

impl A {
    async fn foo(&self, f: &u32) -> &A { self }
}

fn main() { }
