// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=lazy

// NB: We do not expect *any* monomorphization to be generated here.

#![feature(const_fn)]
#![deny(dead_code)]
#![crate_type = "rlib"]

pub const fn foo(x: u32) -> u32 {
    x + 0xf00
}
