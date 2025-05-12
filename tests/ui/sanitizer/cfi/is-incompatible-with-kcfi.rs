// Verifies that `-Zsanitizer=cfi` is incompatible with `-Zsanitizer=kcfi`.
//
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer=kcfi

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR cfi sanitizer is not supported for this target
//~? ERROR `-Zsanitizer=cfi` is incompatible with `-Zsanitizer=kcfi`
