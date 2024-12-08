//@ revisions: x86 x86_64 aarch64

//@ compile-flags: -Zfunction-return=thunk-extern

//@[x86] check-pass
//@[x86] needs-llvm-components: x86
//@[x86] compile-flags: --target i686-unknown-linux-gnu

//@[x86_64] check-pass
//@[x86_64] needs-llvm-components: x86
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu

//@[aarch64] check-fail
//@[aarch64] needs-llvm-components: aarch64
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] error-pattern: `-Zfunction-return` (except `keep`) is only supported on x86 and x86_64

#![feature(no_core)]
#![no_core]
#![no_main]
