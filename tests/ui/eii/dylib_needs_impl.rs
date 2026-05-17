//@ no-prefer-dynamic
//@ needs-crate-type: dylib
#![crate_type = "dylib"]
#![feature(extern_item_impls)]

#[eii(eii1)] //~ ERROR `#[eii1]` required, but not found
fn decl1(x: u64);
