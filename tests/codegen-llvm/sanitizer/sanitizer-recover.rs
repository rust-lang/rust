// Verifies that AddressSanitizer and MemorySanitizer
// recovery mode can be enabled with -Zsanitizer-recover.
//
//@ needs-sanitizer-address
//@ needs-sanitizer-memory
//@ revisions:ASAN ASAN-RECOVER MSAN MSAN-RECOVER MSAN-RECOVER-LTO
//@ no-prefer-dynamic
//@                   compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
//@                   compile-flags: -Ctarget-feature=-crt-static
//@[ASAN]             compile-flags: -Zsanitizer=address -Copt-level=0
//@[ASAN-RECOVER]     compile-flags: -Zsanitizer=address -Zsanitizer-recover=address -Copt-level=0
//@[MSAN]             compile-flags: -Zsanitizer=memory
//@[MSAN-RECOVER]     compile-flags: -Zsanitizer=memory  -Zsanitizer-recover=memory
//@[MSAN-RECOVER-LTO] compile-flags: -Zsanitizer=memory  -Zsanitizer-recover=memory -C lto=fat
//
// MSAN-NOT:         @__msan_keep_going
// MSAN-RECOVER:     @__msan_keep_going = weak_odr {{.*}}constant i32 1
// MSAN-RECOVER-LTO: @__msan_keep_going = weak_odr {{.*}}constant i32 1

// ASAN-LABEL: define dso_local i32 @penguin(
// ASAN:         call void @__asan_report_load4(i64 %0)
// ASAN:         unreachable
// ASAN:       }
//
// ASAN-RECOVER-LABEL: define dso_local i32 @penguin(
// ASAN-RECOVER:         call void @__asan_report_load4_noabort(
// ASAN-RECOVER-NOT:     unreachable
// ASAN:               }
//
// MSAN-LABEL: define dso_local noundef i32 @penguin(
// MSAN:         call void @__msan_warning{{(_with_origin_noreturn\(i32 0\)|_noreturn\(\))}}
// MSAN:         unreachable
// MSAN:       }
//
// MSAN-RECOVER-LABEL: define dso_local noundef i32 @penguin(
// MSAN-RECOVER:         call void @__msan_warning{{(_with_origin\(i32 0\)|\(\))}}
// MSAN-RECOVER-NOT:     unreachable
// MSAN-RECOVER:       }
//
// MSAN-RECOVER-LTO-LABEL: define dso_local noundef i32 @penguin(
// MSAN-RECOVER-LTO:          call void @__msan_warning{{(_with_origin\(i32 0\)|\(\))}}
// MSAN-RECOVER-LTO-NOT:      unreachable
// MSAN-RECOVER-LTO:       }
//
#[no_mangle]
pub fn penguin(p: &mut i32) -> i32 {
    *p
}

fn main() {}
