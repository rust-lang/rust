//@ revisions: aarch64_gl i686_g i686_gl i686_uwp_g x86_64_g x86_64_gl x86_64_uwp_g
//@ compile-flags: --crate-type cdylib -Csplit-debuginfo=off
//@ check-pass

//@[aarch64_gl] compile-flags: --target aarch64-pc-windows-gnullvm
//@[aarch64_gl] needs-llvm-components: aarch64

//@[i686_g] compile-flags: --target i686-pc-windows-gnu
//@[i686_g] needs-llvm-components: x86

//@[i686_gl] compile-flags: --target i686-pc-windows-gnullvm
//@[i686_gl] needs-llvm-components: x86

//@[i686_uwp_g] compile-flags: --target i686-uwp-windows-gnu
//@[i686_uwp_g] needs-llvm-components: x86

//@[x86_64_g] compile-flags: --target x86_64-pc-windows-gnu
//@[x86_64_g] needs-llvm-components: x86

//@[x86_64_gl] compile-flags: --target x86_64-pc-windows-gnullvm
//@[x86_64_gl] needs-llvm-components: x86

//@[x86_64_uwp_g] compile-flags: --target x86_64-uwp-windows-gnu
//@[x86_64_uwp_g] needs-llvm-components: x86

#![feature(no_core)]

#![no_core]
#![no_std]
