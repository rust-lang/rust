// This test checks that data layout mismatches emit an error.
//
//@ build-fail
//@ needs-llvm-components: x86
//@ compile-flags: --crate-type=lib --target={{src-base}}/codegen/mismatched-data-layout.json -Z unstable-options
//@ error-pattern: differs from LLVM target's
//@ normalize-stderr-test: "`, `[A-Za-z0-9-:]*`" -> "`, `normalized data layout`"
//@ normalize-stderr-test: "layout, `[A-Za-z0-9-:]*`" -> "layout, `normalized data layout`"

#![feature(lang_items, no_core, auto_traits)]
#![no_core]

#[lang = "sized"]
trait Sized {}
