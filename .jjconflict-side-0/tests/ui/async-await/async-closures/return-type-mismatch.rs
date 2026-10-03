//@ aux-build:block-on.rs
//@ edition:2021

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let x = async || -> i32 { 0 };
        let y: usize = x().await;
        //~^ ERROR mismatched types
    });
}
