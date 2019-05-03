// edition:2018
// run-pass

#![feature(async_await)]

async fn foo(n: u32, mut vec: Vec<u32>) {
    vec.push(n);
}

fn main() {}
