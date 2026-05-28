// Verifies that the `#[sanitize(hwaddress = "off")]` attribute also turns off
// the kernel hardware-assisted address sanitizer.
//
//@ add-minicore
//@ compile-flags: -Zsanitizer=kernel-hwaddress --target aarch64-unknown-none
//@ compile-flags: -Ctarget-feature=-crt-static -Copt-level=0
//@ needs-llvm-components: aarch64

#![crate_type = "rlib"]
#![feature(no_core, sanitize, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-NOT:   sanitize_hwaddress
// CHECK-LABEL: define {{.*}} @unsanitized
// CHECK:       start:
// CHECK-NOT:   call void @llvm.hwasan.check.memaccess
// CHECK:       }
#[sanitize(hwaddress = "off")]
#[no_mangle]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK:       sanitize_hwaddress
// CHECK-LABEL: define {{.*}} @sanitized
// CHECK:       start:
// CHECK:       call void @llvm.hwasan.check.memaccess
// CHECK:       }
#[no_mangle]
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
