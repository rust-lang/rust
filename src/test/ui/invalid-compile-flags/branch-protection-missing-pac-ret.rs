// revisions: BADFLAGS BADTARGET
// [BADFLAGS] compile-flags: --target=aarch64-unknown-linux-gnu -Zbranch-protection=leaf
// [BADFLAGS] check-fail
// [BADFLAGS] needs-llvm-components: aarch64
// [BADTARGET] compile-flags: --target=x86_64-unknown-linux-gnu -Zbranch-protection=bti
// [BADTARGET] build-fail
// [BADTARGET] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
