//@ revisions: by_flag by_feature
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ [by_flag]compile-flags: -Zx86-retpoline
//@ [by_feature]compile-flags: -Ctarget-feature=+retpoline-external-thunk
//@ [by_flag]build-pass
// For now this is just a warning.
//@ [by_feature]build-pass
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

#[lang = "sized"]
pub trait Sized {}
