//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![feature(btf_relocations)]
#![feature(no_core)]
#![no_core]

extern crate minicore;

#[btf_relocatable]
struct ValidStructInner {
    field: u32,
}

#[btf_relocatable]
struct ValidStruct {
    field: u32,
    inner: ValidStructInner,
}

#[btf_relocatable]
union ValidUnion {
    word: u64,
    half: u32,
}

#[btf_relocatable]
//~^ ERROR the `btf_relocatable` attribute cannot be used on enums
enum InvalidEnum {
    A,
}

#[btf_relocatable]
//~^ ERROR the `btf_relocatable` attribute cannot be used on functions
fn invalid_function() {}

#[btf_relocatable]
//~^ ERROR the `btf_relocatable` attribute cannot be used on traits
trait InvalidTrait {}

fn main() {}
