//@ revisions: badflags badtarget
//@ [badflags] compile-flags: --target=aarch64-unknown-linux-gnu -Zbranch-protection=leaf
//@ [badflags] check-fail
//@ [badflags] needs-llvm-components: aarch64
//@ [badtarget] compile-flags: --target=x86_64-unknown-linux-gnu -Zbranch-protection=bti
//@ [badtarget] check-fail
//@ [badtarget] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}
