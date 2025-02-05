// Verifies that `-Csanitize=cfi` is incompatible with `-Csanitize=kcfi`.
//
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zunstable-options -Csanitize=cfi -Csanitize=kcfi

#![feature(no_core)]
#![no_core]
#![no_main]
