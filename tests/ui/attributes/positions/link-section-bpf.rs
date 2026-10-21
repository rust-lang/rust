//! Checks that on BPF, `link_section` can be used on foreign functions and statics:
//! that is how kfuncs and ksyms are placed in ".ksyms". See `link-section.rs` for
//! the warning on other targets.

//@ add-minicore
//@ check-pass
//@ edition:2024
//@ compile-flags: --target bpfel-unknown-none
//@ needs-llvm-components: bpf
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]
#![deny(unused_attributes)]

extern crate minicore;

unsafe extern "C" {
    #[unsafe(link_section = ".ksyms")]
    pub fn bpf_kfunc();
    #[unsafe(link_section = ".ksyms")]
    pub static bpf_ksym: u32;
}
