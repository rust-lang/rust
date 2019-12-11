// Verifies that AddressSanitizer and MemorySanitizer
// recovery mode can be enabled with -Zsanitizer-recover.
//
// needs-sanitizer-support
// only-linux
// only-x86_64
// revisions:ASAN ASAN-RECOVER MSAN MSAN-RECOVER
//
//[ASAN]         compile-flags: -Zsanitizer=address
//[ASAN-RECOVER] compile-flags: -Zsanitizer=address -Zsanitizer-recover=address
//[MSAN]         compile-flags: -Zsanitizer=memory
//[MSAN-RECOVER] compile-flags: -Zsanitizer=memory  -Zsanitizer-recover=memory

#![crate_type="lib"]

// ASAN-LABEL:         define i32 @penguin(
// ASAN-RECOVER-LABEL: define i32 @penguin(
// MSAN-LABEL:         define i32 @penguin(
// MSAN-RECOVER-LABEL: define i32 @penguin(
#[no_mangle]
pub fn penguin(p: &mut i32) -> i32 {
    // ASAN:             call void @__asan_report_load4(i64 %0)
    // ASAN:             unreachable
    //
    // ASAN-RECOVER:     call void @__asan_report_load4_noabort(
    // ASAN-RECOVER-NOT: unreachable
    //
    // MSAN:             call void @__msan_warning_noreturn()
    // MSAN:             unreachable
    //
    // MSAN-RECOVER:     call void @__msan_warning()
    // MSAN-RECOVER-NOT: unreachable
    *p
}
