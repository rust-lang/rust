#![crate_type = "rlib"]

#[eii] //~ ERROR use of unstable library feature `extern_item_impls`
fn hello(x: u64);

#[hello]
fn hello_impl(x: u64) {
    println!("{x:?}")
}

#[eii(hello2)] //~ ERROR use of unstable library feature `extern_item_impls`
static HELLO: u64;

#[hello2]
static HELLO_IMPL: u64 = 2;
