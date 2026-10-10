//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

fn wrapper(f: impl Fn(String)) -> impl AsyncFn(String) {
    async move |s| f(s)
}

fn main() {
    block_on::block_on(async {
        wrapper(|who| println!("Hello, {who}!"))(String::from("world")).await;
    });
}
