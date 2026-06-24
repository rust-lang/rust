//@ needs-crate-type: bin
#![feature(extern_item_impls)]

#[eii(eii1)] //~ ERROR `#[eii1]` function required, but not found
fn decl1(x: u64);

fn main() {}
