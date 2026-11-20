//! The `link_section` attribute is accepted on foreign functions and statics,
//! and BPF kfuncs and ksyms are marked by putting them in ".ksyms".

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
