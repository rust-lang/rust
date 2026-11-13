// Compiler:
//
// Run-time:
//   status: 0

// Checks the `#[linkage]` flavours an `extern` static can be imported with, against the symbols
// `tests/c/import_linkage.c` defines.
//
// The value of such an import is the address of the symbol rather than its contents, which is why
// the types are pointers: an `extern_weak` import of a symbol nobody defines reads as null instead
// of failing the link.

#![feature(linkage, no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

extern "C" {
    #[linkage = "external"]
    static external_value: *const i32;
    #[linkage = "available_externally"]
    static available_externally_value: *const i32;
    #[linkage = "linkonce"]
    static linkonce_value: *const i32;
    #[linkage = "linkonce_odr"]
    static linkonce_odr_value: *const i32;
    #[linkage = "weak"]
    static weak_value: *const i32;
    #[linkage = "weak_odr"]
    static weak_odr_value: *const i32;
    #[linkage = "common"]
    static common_value: *const i32;
    #[linkage = "extern_weak"]
    static extern_weak_value: *const i32;
    // An import is an undefined reference whatever the flavour says. Upstream bug: rustc lowers
    // this one to an internal declaration, which LLVM's verifier rejects ("Global is external, but
    // doesn't have external or weak linkage!") and which crashes cg_llvm at -O3.
    #[linkage = "internal"]
    static internal_value: *const i32;

    // Nothing defines this one, so it stays null instead of breaking the link.
    #[linkage = "extern_weak"]
    static undefined_value: *const i32;
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        if *external_value != 1 {
            return 1;
        }
        if *available_externally_value != 2 {
            return 2;
        }
        if *linkonce_value != 3 {
            return 3;
        }
        if *linkonce_odr_value != 4 {
            return 4;
        }
        if *weak_value != 5 {
            return 5;
        }
        if *weak_odr_value != 6 {
            return 6;
        }
        if *common_value != 7 {
            return 7;
        }
        if *extern_weak_value != 8 {
            return 8;
        }
        if *internal_value != 9 {
            return 9;
        }
        if undefined_value as usize != 0 {
            return 10;
        }
    }
    0
}
