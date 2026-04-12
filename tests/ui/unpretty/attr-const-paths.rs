//@ compile-flags: -Zunpretty=hir
//@ check-pass

#![feature(const_attr_paths)]

const ALIGN: usize = 8;

#[repr(align(ALIGN))]
struct Aligned;

#[repr(packed(2))]
struct Packed(u32);

fn main() {}
