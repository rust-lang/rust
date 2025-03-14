//@ add-core-stubs
//@ compile-flags: -Copt-level=3

//@ revisions:x86_64 i686 aarch64-apple aarch64-windows aarch64-linux arm riscv

//@[x86_64] compile-flags: --target x86_64-unknown-uefi
//@[x86_64] needs-llvm-components: x86
//@[i686] compile-flags: --target i686-unknown-linux-musl
//@[i686] needs-llvm-components: x86
//@[aarch64-windows] compile-flags: --target aarch64-pc-windows-msvc
//@[aarch64-windows] needs-llvm-components: aarch64
//@[aarch64-linux] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64-linux] needs-llvm-components: aarch64
//@[aarch64-apple] compile-flags: --target aarch64-apple-darwin
//@[aarch64-apple] needs-llvm-components: aarch64
//@[arm] compile-flags: --target armv7r-none-eabi
//@[arm] needs-llvm-components: arm
//@[riscv] compile-flags: --target riscv64gc-unknown-none-elf
//@[riscv] needs-llvm-components: riscv

// See bottom of file for a corresponding C source file that is meant to yield
// equivalent declarations.
#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(i8)]
pub enum Type {
    Type1 = 0,
    Type2 = 1,
}

// To accommodate rust#97800, one might consider writing the below as:
//
// `define{{( dso_local)?}} noundef{{( signext)?}} i8 @test()`
//
// but based on rust#80556, it seems important to actually check for the
// presence of the `signext` for those targets where we expect it.

// CHECK: define{{( dso_local)?}} noundef
// x86_64-SAME:                 signext
// aarch64-apple-SAME:          signext
// aarch64-windows-NOT: signext
// aarch64-linux-NOT:   signext
// arm-SAME:                    signext
// riscv-SAME:                  signext
// CHECK-SAME: i8 @test()

#[no_mangle]
pub extern "C" fn test() -> Type {
    Type::Type1
}
