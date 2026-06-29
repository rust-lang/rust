// Verifies that AddressSanitizer and MemorySanitizer
// recovery mode can be enabled with -Zsanitizer-recover.
//
//@ needs-sanitizer-address
//@ needs-sanitizer-memory
//@ revisions:ASAN ASAN-RECOVER MSAN MSAN-RECOVER MSAN-RECOVER-LTO
//@ no-prefer-dynamic
//@                   compile-flags: -Cunsafe-allow-abi-mismatch=sanitizer
//@                   compile-flags: -Ctarget-feature=-crt-static
//@[ASAN]             compile-flags: -Csanitizer=address -Copt-level=0 -Zunstable-options
//@[ASAN-RECOVER]     compile-flags: -Csanitizer=address -Zsanitizer-recover=address -Copt-level=0
//@[ASAN-RECOVER]     compile-flags: -Zunstable-options
//@[MSAN]             compile-flags: -Tsanitizer=memory -Zunstable-options
//@[MSAN-RECOVER]     compile-flags: -Tsanitizer=memory -Zsanitizer-recover=memory
//@[MSAN-RECOVER]     compile-flags: -Zunstable-options
//@[MSAN-RECOVER-LTO] compile-flags: -Tsanitizer=memory -Zsanitizer-recover=memory -C lto=fat
//@[MSAN-RECOVER-LTO] compile-flags: -Zunstable-options
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
