//@ only-apple
//@ ignore-backends: gcc

#![feature(extern_item_impls)]
#![crate_type = "lib"]
#[eii(eii1)]
pub static DECL1: u64 = 5;
//~^ ERROR `#[eii]` cannot be used on statics with a value on Apple targets
