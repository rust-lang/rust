#![feature(optimize_attribute)]
#![feature(stmt_expr_attributes)]
#![deny(unused_attributes)]
#![allow(dead_code)]

//@ edition: 2018

#[optimize(speed)] //~ ERROR attribute cannot be used on
struct F;

fn invalid() {
    #[optimize(speed)] //~ ERROR attribute cannot be used on
    {
        1
    };
}

#[optimize(speed)]
fn valid() {}

#[optimize(speed)] //~ ERROR attribute cannot be used on
mod valid_module {}

#[optimize(speed)] //~ ERROR attribute cannot be used on
impl F {}

fn main() {
    let _ = #[optimize(speed)]
    (|| 1);
}

use std::future::Future;

fn async_block() -> impl Future<Output = ()> {
    #[optimize(speed)]
    async { }
}

#[optimize(speed)]
async fn async_fn() {
    ()
}
