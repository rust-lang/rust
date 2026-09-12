//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    #[unsafe(link_section = ".ksyms")]
    //~^ ERROR `link_section` on foreign items is only supported on BPF targets
    static foo: u32;

    #[unsafe(link_section = ".ksyms")]
    //~^ ERROR `link_section` on foreign items is only supported on BPF targets
    fn bar();
}

#[unsafe(link_section = ".text")]
fn regular_fn() {}

#[unsafe(link_section = ".data")]
static BAZ: u32 = 42;
