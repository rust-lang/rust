//@ no-prefer-dynamic
//@ aux-build: decl.rs
#![crate_type = "rlib"]
#![feature(eii)]

extern crate decl;


#[unsafe(decl::eii1)] //~ ERROR multiple implementations of `#[eii1]`
fn other(x: u64) {
    println!("1{x}");
}
