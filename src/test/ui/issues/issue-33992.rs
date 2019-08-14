// run-pass
// ignore-windows
// ignore-macos
// ignore-wasm32-bare common linkage not implemented right now

#![feature(linkage)]

#[linkage = "common"]
pub static mut TEST1: u32 = 0u32;

#[linkage = "external"]
pub static TEST2: bool = true;

#[linkage = "internal"]
pub static TEST3: bool = true;

#[linkage = "linkonce"]
pub static TEST4: bool = true;

#[linkage = "linkonce_odr"]
pub static TEST5: bool = true;

#[linkage = "private"]
pub static TEST6: bool = true;

#[linkage = "weak"]
pub static TEST7: bool = true;

#[linkage = "weak_odr"]
pub static TEST8: bool = true;

fn main() {}
