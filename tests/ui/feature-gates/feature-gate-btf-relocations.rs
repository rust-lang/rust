//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![feature(no_core)]
#![no_core]

extern crate minicore;

#[btf_relocatable]
//~^ ERROR the `btf_relocatable` attribute is an experimental feature
struct KernelType {
    field: u32,
}

fn main() {}
