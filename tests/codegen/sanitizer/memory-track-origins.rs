// Verifies that MemorySanitizer track-origins level can be controlled
// with -Zsanitizer-memory-track-origins option.
//
//@ needs-sanitizer-memory
//@ revisions: msan-0 msan-1 msan-2 msan-1-lto msan-2-lto
//
//@ compile-flags: -Zsanitizer=memory -Ctarget-feature=-crt-static
//@[msan-0] compile-flags:
//@[msan-1] compile-flags: -Zsanitizer-memory-track-origins=1
//@[msan-2] compile-flags: -Zsanitizer-memory-track-origins
//@[msan-1-lto] compile-flags: -Zsanitizer-memory-track-origins=1 -C lto=fat
//@[msan-2-lto] compile-flags: -Zsanitizer-memory-track-origins -C lto=fat

#![crate_type = "lib"]

// CHECK-MSAN-0-NOT: @__msan_track_origins
// CHECK-MSAN-1:     @__msan_track_origins = weak_odr {{.*}}constant i32 1
// CHECK-MSAN-2:     @__msan_track_origins = weak_odr {{.*}}constant i32 2
// CHECK-MSAN-1-LTO: @__msan_track_origins = weak_odr {{.*}}constant i32 1
// CHECK-MSAN-2-LTO: @__msan_track_origins = weak_odr {{.*}}constant i32 2
//
// CHECK-MSAN-0-LABEL: define void @copy(
// CHECK-MSAN-1-LABEL: define void @copy(
// CHECK-MSAN-2-LABEL: define void @copy(
#[no_mangle]
pub fn copy(dst: &mut i32, src: &i32) {
    // CHECK-MSAN-0-NOT: call i32 @__msan_chain_origin(
    // CHECK-MSAN-1-NOT: call i32 @__msan_chain_origin(
    // CHECK-MSAN-2:     call i32 @__msan_chain_origin(
    *dst = *src;
}
