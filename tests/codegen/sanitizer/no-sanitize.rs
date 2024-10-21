// Verifies that no_sanitize attribute can be used to
// selectively disable sanitizer instrumentation.
//
//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Ctarget-feature=-crt-static -Copt-level=0

#![crate_type = "lib"]
#![feature(no_sanitize)]

// CHECK:     @UNSANITIZED = constant{{.*}} no_sanitize_address
// CHECK-NOT: @__asan_global_UNSANITIZED
#[no_mangle]
#[no_sanitize(address)]
pub static UNSANITIZED: u32 = 0;

// CHECK: @__asan_global_SANITIZED
#[no_mangle]
pub static SANITIZED: u32 = 0;

// CHECK-LABEL: ; no_sanitize::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[no_sanitize(address)]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; no_sanitize::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
