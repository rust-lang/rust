//@ revisions: e2024 none
//@[e2024] edition: 2024

async gen fn foo() {}
//[none]~^ ERROR: `async fn` is not permitted in Rust 2015
//[none]~| ERROR: expected one of `extern`, `fn`, `safe`, or `unsafe`, found `gen`
//[e2024]~^^^ ERROR: gen blocks are experimental

fn main() {}
