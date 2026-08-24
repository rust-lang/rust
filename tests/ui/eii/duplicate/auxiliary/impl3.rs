//@ aux-build: decl.rs
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

extern crate decl;


#[decl::eii1]
fn other(x: u64) {
    println!("3{x}");
}
