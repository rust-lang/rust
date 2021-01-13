// Checks if the correct annotation for the efiapi ABI is passed to llvm.

// revisions:x86_64 i686 aarch64 arm riscv
// needs-llvm-components: aarch64 arm riscv

//[x86_64] compile-flags: --target x86_64-unknown-uefi
//[i686] compile-flags: --target i686-unknown-linux-musl
//[aarch64] compile-flags: --target aarch64-unknown-none
//[arm] compile-flags: --target armv7r-none-eabi
//[riscv] compile-flags: --target riscv64gc-unknown-none-elf
// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(no_core, lang_items, abi_efiapi)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="freeze"]
trait Freeze { }
#[lang="copy"]
trait Copy { }

//x86_64: define win64cc void @has_efiapi
//i686: define void @has_efiapi
//aarch64: define void @has_efiapi
//arm: define void @has_efiapi
//riscv: define void @has_efiapi
#[no_mangle]
pub extern "efiapi" fn has_efiapi() {}
