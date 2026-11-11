//@ aux-build:block-on.rs
//@ edition:2021

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let c = async |x| {};
        c(1i32).await;
        c(2usize).await;
        //~^ ERROR mismatched types
    });
}
