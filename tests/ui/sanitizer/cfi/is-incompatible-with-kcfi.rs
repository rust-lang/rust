// Verifies that `-Csanitize=cfi` is incompatible with `-Csanitize=kcfi`.
//
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi -Csanitize=kcfi
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR cfi sanitizer is not supported for this target
//~? ERROR `-Csanitize=cfi` is incompatible with `-Csanitize=kcfi`
