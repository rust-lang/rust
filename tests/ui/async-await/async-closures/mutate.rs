//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let mut prefix = String::from("Hello");
        let mut c = async move |x: &str| {
            prefix.push(',');
            println!("{prefix} {x}!")
        };
        c("world").await;
        c("rust").await;
    });
}
