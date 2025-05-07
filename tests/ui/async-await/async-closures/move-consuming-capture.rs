//@ aux-build:block-on.rs
//@ edition:2021

extern crate block_on;

struct NoCopy;

fn main() {
    block_on::block_on(async {
        let s = NoCopy;
        let x = async move || {
            drop(s);
        };
        x().await;
        x().await;
        //~^ ERROR use of moved value: `x`
    });
}
