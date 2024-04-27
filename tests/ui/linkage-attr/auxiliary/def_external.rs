#![feature(linkage)]
#![crate_type = "lib"]

#[linkage="external"]
pub static EXTERN: u32 = 0;
