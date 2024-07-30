// Verifies that AddressSanitizer and MemorySanitizer
// recovery mode can be enabled with -Zsanitizer-recover.
//
//@ needs-sanitizer-address
//@ needs-sanitizer-memory
//@ revisions: asan asan-recover msan msan-recover msan-recover-lto
//@ no-prefer-dynamic
//
//@                   compile-flags: -Ctarget-feature=-crt-static
//@[asan]             compile-flags: -Zsanitizer=address -Copt-level=0
//@[asan-recover]     compile-flags: -Zsanitizer=address -Zsanitizer-recover=address -Copt-level=0
//@[msan]             compile-flags: -Zsanitizer=memory
//@[msan-recover]     compile-flags: -Zsanitizer=memory  -Zsanitizer-recover=memory
//@[msan-recover-lto] compile-flags: -Zsanitizer=memory  -Zsanitizer-recover=memory -C lto=fat
//
// CHECK-MSAN-NOT:         @__msan_keep_going
// CHECK-MSAN-RECOVER:     @__msan_keep_going = weak_odr {{.*}}constant i32 1
// CHECK-MSAN-RECOVER-LTO: @__msan_keep_going = weak_odr {{.*}}constant i32 1

// CHECK-ASAN-LABEL: define dso_local i32 @penguin(
// CHECK-ASAN:         call void @__asan_report_load4(i64 %0)
// CHECK-ASAN:         unreachable
// CHECK-ASAN:       }
//
// CHECK-ASAN-RECOVER-LABEL: define dso_local i32 @penguin(
// CHECK-ASAN-RECOVER:         call void @__asan_report_load4_noabort(
// CHECK-ASAN-RECOVER-NOT:     unreachable
// CHECK-ASAN:               }
//
// CHECK-MSAN-LABEL: define dso_local noundef i32 @penguin(
// CHECK-MSAN:         call void @__msan_warning{{(_with_origin_noreturn\(i32 0\)|_noreturn\(\))}}
// CHECK-MSAN:         unreachable
// CHECK-MSAN:       }
//
// CHECK-MSAN-RECOVER-LABEL: define dso_local noundef i32 @penguin(
// CHECK-MSAN-RECOVER:         call void @__msan_warning{{(_with_origin\(i32 0\)|\(\))}}
// CHECK-MSAN-RECOVER-NOT:     unreachable
// CHECK-MSAN-RECOVER:       }
//
// CHECK-MSAN-RECOVER-LTO-LABEL: define dso_local noundef i32 @penguin(
// CHECK-MSAN-RECOVER-LTO:          call void @__msan_warning{{(_with_origin\(i32 0\)|\(\))}}
// CHECK-MSAN-RECOVER-LTO-NOT:      unreachable
// CHECK-MSAN-RECOVER-LTO:       }
//
#[no_mangle]
pub fn penguin(p: &mut i32) -> i32 {
    *p
}

fn main() {}
