#![feature(coverage_attribute)]
//@ edition: 2018

//@ aux-build: executor.rs
extern crate executor;

fn non_async_func() {
    println!("non_async_func was covered");
    let b = true;
    if b {
        println!("non_async_func println in block");
    }
}

async fn async_func() {
    println!("async_func was covered");
    let b = true;
    if b {
        println!("async_func println in block");
    }
}

async fn async_func_just_println() {
    println!("async_func_just_println was covered");
}

fn main() {
    println!("codecovsample::main");

    non_async_func();

    executor::block_on(async_func());
    executor::block_on(async_func_just_println());
}
