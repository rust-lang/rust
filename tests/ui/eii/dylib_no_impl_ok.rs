//@ needs-crate-type: dylib
//@ check-pass
#![crate_type = "dylib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
fn decl1(x: u64);
