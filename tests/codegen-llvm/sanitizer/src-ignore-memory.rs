//@ revisions: ASAN MSAN TSAN HWASAN SAFESTACK
//@[ASAN] needs-sanitizer-address
//@[MSAN] needs-sanitizer-memory
//@[TSAN] needs-sanitizer-thread
//@[HWASAN] needs-sanitizer-hwaddress
//@[SAFESTACK] needs-sanitizer-safestack
//@ compile-flags: -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt -Cunsafe-allow-abi-mismatch=sanitizer
//@ [ASAN] compile-flags: -Zsanitizer=address
//@ [MSAN] compile-flags: -Zsanitizer=memory
//@ [TSAN] compile-flags: -Zsanitizer=thread
//@ [HWASAN] compile-flags: -Zsanitizer=hwaddress -C target-feature=+tagged-globals
//@ [SAFESTACK] compile-flags: -Zsanitizer=safestack

#![crate_type = "lib"]

// CHECK: ; Function Attrs:
// ASAN-NOT: sanitize_address
// MSAN-NOT: sanitize_memory
// TSAN-NOT: sanitize_thread
// HWASAN-NOT: sanitize_hwaddress
// SAFESTACK-NOT: safestack
// CHECK-NEXT: define void @test_file_address
#[no_mangle]
pub fn test_file_address(x: &mut i32) {
    *x = 1;
}

// CHECK: ; Function Attrs:
// ASAN-NOT: sanitize_address
// MSAN-NOT: sanitize_memory
// TSAN-NOT: sanitize_thread
// HWASAN-NOT: sanitize_hwaddress
// SAFESTACK-NOT: safestack
// CHECK-NEXT: define void @test_file_memory
#[no_mangle]
pub fn test_file_memory(x: &mut i32) {
    *x = 2;
}

// CHECK: ; Function Attrs:
// ASAN-NOT: sanitize_address
// MSAN-NOT: sanitize_memory
// TSAN-NOT: sanitize_thread
// HWASAN-NOT: sanitize_hwaddress
// SAFESTACK-NOT: safestack
// CHECK-NEXT: define void @test_file_thread
#[no_mangle]
pub fn test_file_thread(x: &mut i32) {
    *x = 3;
}

// CHECK: ; Function Attrs:
// ASAN-NOT: sanitize_address
// MSAN-NOT: sanitize_memory
// TSAN-NOT: sanitize_thread
// HWASAN-NOT: sanitize_hwaddress
// SAFESTACK-NOT: safestack
// CHECK-NEXT: define void @test_file_hwaddress
#[no_mangle]
pub fn test_file_hwaddress(x: &mut i32) {
    *x = 4;
}

// CHECK: ; Function Attrs:
// ASAN-NOT: sanitize_address
// MSAN-NOT: sanitize_memory
// TSAN-NOT: sanitize_thread
// HWASAN-NOT: sanitize_hwaddress
// SAFESTACK-NOT: safestack
// CHECK-NEXT: define void @test_file_safestack
#[no_mangle]
pub fn test_file_safestack(x: &mut i32) {
    *x = 5;
}
