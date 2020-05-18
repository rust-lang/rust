// Checks whether UEFI targets cross-compile successfully.
//
// This test contains a simple UEFI program that simply exits with return code
// 0. It can be easily run from the UEFI shell (but any other UEFI environment
// works as well). This program is not run as part of the test. The test merely
// verifies the cross-compilation does not fail and an entry-point is emitted.
//
// The imported definitions from the UEFI specification are intentionally left
// incomplete. Only the bits that are actually used by this test are defined.

// min-llvm-version 9.0

// compile-flags: --target x86_64-unknown-uefi

#![feature(abi_efiapi, lang_items, no_core)]
#![no_core]
#![no_main]

#[lang = "sized"]
pub trait Sized {}
#[lang = "freeze"]
pub trait Freeze {}
#[lang = "copy"]
pub trait Copy {}

// CHECK: define win64cc i64 @efi_main{{.*}}
#[export_name = "efi_main"]
pub extern "efiapi" fn main(_h: *mut usize, _st: *mut usize) -> usize {
    return 0;
}
