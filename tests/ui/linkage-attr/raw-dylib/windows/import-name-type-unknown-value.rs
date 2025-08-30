//@ add-core-stubs
//@ compile-flags: --target i686-pc-windows-msvc
//@ needs-llvm-components: x86
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]
#![crate_type = "lib"]

#[link(name = "foo", kind = "raw-dylib", import_name_type = "unknown")]
//~^ ERROR malformed
extern "C" { }
