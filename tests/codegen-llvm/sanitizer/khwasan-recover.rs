// Verifies that KernelHWAddressSanitizer recovery mode can be enabled
// with -Zsanitizer-recover=kernel-hwaddress.
//
//@ add-minicore
//@[KHWASAN] needs-llvm-components: aarch64
//@[KHWASAN-RECOVER] needs-llvm-components: aarch64
//@ revisions: KHWASAN KHWASAN-RECOVER
//@ no-prefer-dynamic
//@ compile-flags: -Copt-level=0
//@[KHWASAN]            compile-flags: -Zsanitizer=kernel-hwaddress --target aarch64-unknown-none
//@[KHWASAN-RECOVER]    compile-flags: -Zsanitizer=kernel-hwaddress
//@[KHWASAN-RECOVER]    compile-flags: -Zsanitizer-recover=kernel-hwaddress
//@[KHWASAN-RECOVER]    compile-flags: --target aarch64-unknown-none

#![feature(no_core, sanitize, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// KHWASAN-LABEL: define{{.*}}@penguin(
// KHWASAN:         call void @llvm.hwasan.check.memaccess
// KHWASAN:         ret i32
// KHWASAN:       }
// KHWASAN:       declare void @__hwasan_load4(i64)

// KHWASAN-RECOVER-LABEL: define{{.*}}@penguin(
// KHWASAN-RECOVER:         call void asm sideeffect "brk #2338", "{x0}"(i64 %{{[0-9]+}})
// KHWASAN-RECOVER-NOT:     unreachable
// KHWASAN-RECOVER:       }
// KHWASAN-RECOVER:       declare void @__hwasan_load4_noabort(i64)

#[no_mangle]
pub unsafe fn penguin(p: *mut i32) -> i32 {
    *p
}
