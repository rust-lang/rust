// Checks if the correct annotation for the efiapi ABI is passed to llvm.

//@ add-core-stubs
//@ revisions:x86_64 i686 aarch64 arm riscv
//@[x86_64] compile-flags: --target x86_64-unknown-uefi
//@[x86_64] filecheck-flags: --check-prefixes=x86_64
//@[x86_64] needs-llvm-components: x86
//@[i686] compile-flags: --target i686-unknown-linux-musl
//@[i686] filecheck-flags: --check-prefixes=i686
//@[i686] needs-llvm-components: x86
//@[aarch64] compile-flags: --target aarch64-unknown-none
//@[aarch64] filecheck-flags: --check-prefixes=aarch64
//@[aarch64] needs-llvm-components: aarch64
//@[arm] compile-flags: --target armv7r-none-eabi
//@[arm] filecheck-flags: --check-prefixes=arm
//@[arm] needs-llvm-components: arm
//@[riscv] compile-flags: --target riscv64gc-unknown-none-elf
//@[riscv] filecheck-flags: --check-prefixes=riscv
//@[riscv] needs-llvm-components: riscv
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

//x86_64: define win64cc void @has_efiapi
//i686: define void @has_efiapi
//aarch64: define dso_local void @has_efiapi
//arm: define dso_local arm_aapcscc void @has_efiapi
//riscv: define dso_local void @has_efiapi
#[no_mangle]
pub extern "efiapi" fn has_efiapi() {}
