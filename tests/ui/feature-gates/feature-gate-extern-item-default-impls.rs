#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii] //~ ERROR eii with default impl is unstable
fn hello(x: u64) {}

#[hello]
fn hello_impl(x: u64) {
    println!("{x:?}")
}

#[eii(hello2)] //~ ERROR eii with default impl is unstable
static HELLO: u64 = 1;

#[hello2]
static HELLO_IMPL: u64 = 2;
