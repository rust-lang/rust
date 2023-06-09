// Verifies that no_sanitize attribute can be used to
// selectively disable sanitizer instrumentation.
//
// needs-sanitizer-address
// compile-flags: -Zsanitizer=address

#![crate_type="lib"]
#![feature(no_sanitize)]

// CHECK-LABEL: ; sanitizer_no_sanitize::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[no_sanitize(address)]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; sanitizer_no_sanitize::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
