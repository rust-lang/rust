//@ add-core-stubs
//@ compile-flags: --target aarch64-pc-windows-msvc
//@ needs-llvm-components: aarch64
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]
#![crate_type = "lib"]

#[link(name = "foo", kind = "raw-dylib", import_name_type = "decorated")]
//~^ ERROR import name type is only supported on x86
extern "C" { }
