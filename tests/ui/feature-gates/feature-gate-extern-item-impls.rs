#![crate_type = "rlib"]

#[eii] //~ ERROR use of unstable library feature `extern_item_impls`
fn hello(x: u64);

#[hello]
fn hello_impl(x: u64) {
    println!("{x:?}")
}
