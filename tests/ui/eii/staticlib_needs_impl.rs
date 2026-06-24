//@ needs-crate-type: staticlib
#![crate_type = "staticlib"]
#![feature(extern_item_impls)]

#[eii(eii1)] //~ ERROR `#[eii1]` function required, but not found
fn decl1(x: u64);
