//@ build-pass

#![crate_type = "lib"]
#![feature(linkage)]

// MergeFunctions will merge these via an anonymous internal
// backing function, which must be named if ThinLTO buffers are used

#[linkage = "weak"]
pub fn fn1(a: u32, b: u32, c: u32) -> u32 {
    a + b + c
}

#[linkage = "weak"]
pub fn fn2(a: u32, b: u32, c: u32) -> u32 {
    a + b + c
}
