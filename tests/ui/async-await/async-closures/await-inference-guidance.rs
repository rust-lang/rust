//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let x = async |x: &str| -> String { x.to_owned() };
        let mut s = x("hello, world").await;
        s.truncate(4);
        println!("{s}");
    });
}
