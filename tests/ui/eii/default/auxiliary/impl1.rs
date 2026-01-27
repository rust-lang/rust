//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

extern crate decl_with_default as decl;


#[unsafe(decl::eii1)] //~ ERROR multiple implementations of `#[eii1]`
fn other(x: u64) {
    println!("1{x}");
}
