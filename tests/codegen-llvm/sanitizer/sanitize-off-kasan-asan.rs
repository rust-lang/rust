// Verifies that the `#[sanitize(kernel_address = "off")]` attribute also turns off
// the address sanitizer.
//
//@ needs-sanitizer-address
//@ compile-flags: -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=address

#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK-LABEL: ; sanitize_off_kasan_asan::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[sanitize(kernel_address = "off")]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; sanitize_off_kasan_asan::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
