//@ add-core-stubs
//@ compile-flags: --target i686-pc-windows-msvc
//@ needs-llvm-components: x86
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]
#![crate_type = "lib"]

#[link(name = "foo", import_name_type = "decorated")]
//~^ ERROR import name type can only be used with link kind `raw-dylib`
extern "C" { }

#[link(name = "bar", kind = "static", import_name_type = "decorated")]
//~^ ERROR import name type can only be used with link kind `raw-dylib`
extern "C" { }

// Specifying `import_name_type` before `kind` shouldn't raise an error.
#[link(name = "bar", import_name_type = "decorated", kind = "raw-dylib")]
extern "C" { }
