#![allow(warnings)]
#![crate_name = "a"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
#[inline(never)]
pub fn foo(b: u8) -> u32 {
    b as u32
}

#[cfg(rpass2)]
#[inline(never)]
pub fn foo(b: u8) -> u32 {
    (b + 42) as u32
}

pub fn bar(b: u8) -> u32 {
    bar_impl(b) as u32
}

#[cfg(rpass1)]
#[inline(never)]
fn bar_impl(b: u8) -> u16 {
    b as u16
}

#[cfg(rpass2)]
#[inline(never)]
fn bar_impl(b: u8) -> u32 {
    (b + 42) as u32
}
