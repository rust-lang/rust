// Verifies that KernelAddressSanitizer recovery mode can be enabled
// with -Zsanitizer-recover=kernel-address.
//
//@ add-minicore
//@ revisions: KASAN KASAN-RECOVER
//@ compile-flags: -Copt-level=0
//@ needs-llvm-components: x86
//@ compile-flags: -Zsanitizer=kernel-address --target x86_64-unknown-none
//@[KASAN-RECOVER] compile-flags: -Zsanitizer-recover=kernel-address

#![feature(no_core, sanitize, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// KASAN-LABEL: define{{.*}}@penguin(
// KASAN:         call void @__asan_report_load4(
// KASAN:         unreachable
// KASAN:       }

// KASAN-RECOVER-LABEL: define{{.*}}@penguin(
// KASAN-RECOVER:         call void @__asan_report_load4_noabort(
// KASAN-RECOVER-NOT:     unreachable
// KASAN-RECOVER:       }

#[no_mangle]
pub unsafe fn penguin(p: *mut i32) -> i32 {
    *p
}
