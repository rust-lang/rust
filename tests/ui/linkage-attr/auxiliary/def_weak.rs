#![feature(linkage)]
#![crate_type = "lib"]

#[linkage = "weak"]
pub static WEAK: u32 = 0;
