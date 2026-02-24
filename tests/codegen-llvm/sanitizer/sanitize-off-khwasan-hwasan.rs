// Verifies that the `#[sanitize(kernel_hwaddress = "off")]` attribute also turns off
// the hardware-assisted address sanitizer.
//
//@ needs-sanitizer-hwaddress
//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitizer
//@ compile-flags: -Ctarget-feature=-crt-static
//@ compile-flags: -Zsanitizer=hwaddress -Copt-level=0

#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK-NOT:   sanitize_hwaddress
// CHECK-LABEL: define {{.*}} @unsanitized
// CHECK:       start:
// CHECK-NOT:   call void @llvm.hwasan.check.memaccess
// CHECK:       }
#[sanitize(kernel_hwaddress = "off")]
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
