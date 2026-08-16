//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![feature(btf_relocations)]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[btf_relocatable]
#[repr(C)]
struct Inner {
    value: u32,
}

#[btf_relocatable]
#[repr(C)]
struct Outer {
    inner: Inner,
}

fn direct(inner: &Inner) -> u32 {
    inner.value
    //~^ ERROR cannot access fields of a `btf_relocatable` type directly
}

fn nested(outer: &Outer) -> u32 {
    outer.inner.value
    //~^ ERROR cannot access fields of a `btf_relocatable` type directly
}

fn offset() -> usize {
    mem::offset_of!(Inner, value)
    //~^ ERROR cannot use `offset_of!` with a `btf_relocatable` type
}

fn nested_offset() -> usize {
    mem::offset_of!(Outer, inner.value)
    //~^ ERROR cannot use `offset_of!` with a `btf_relocatable` type
}

fn main() {}
