//@ aux-build:block-on.rs
//@ edition:2021
//@ build-pass

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let x = async move |x: &str| {
            println!("{x}");
        };
        x("hello!").await;
    });
}
