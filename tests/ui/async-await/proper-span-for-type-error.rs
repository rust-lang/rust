//@ edition:2021
//@ run-rustfix
#![allow(dead_code)]

async fn a() {}

async fn foo() -> Result<(), i32> {
    a().await //~ ERROR mismatched types
}

fn main() {}
