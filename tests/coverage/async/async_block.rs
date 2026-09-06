#![feature(coverage_attribute)]
//@ edition: 2021

//@ aux-build: executor.rs
extern crate executor;

fn main() {
    for i in 0..16 {
        let future = async {
            if i >= 12 {
                println!("big");
            } else {
                println!("small");
            }
        };
        executor::block_on(future);
    }
}
