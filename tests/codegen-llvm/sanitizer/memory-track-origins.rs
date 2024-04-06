// Verifies that MemorySanitizer track-origins level can be controlled
// with -Zsanitizer-memory-track-origins option.
//
//@ needs-sanitizer-memory
//@ revisions:MSAN-0 MSAN-1 MSAN-2 MSAN-1-LTO MSAN-2-LTO
//@[MSAN-0] compile-flags: -Zunstable-options -Csanitize=memory
//@[MSAN-1] compile-flags: -Zunstable-options -Csanitize=memory -Zsanitizer-memory-track-origins=1
//@[MSAN-2] compile-flags: -Zunstable-options -Csanitize=memory -Zsanitizer-memory-track-origins
//@[MSAN-1-LTO] compile-flags: -Clto=fat -Zunstable-options -Csanitize=memory -Zsanitizer-memory-track-origins=1
//@[MSAN-2-LTO] compile-flags: -Clto=fat -Zunstable-options -Csanitize=memory -Zsanitizer-memory-track-origins
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize

#![crate_type = "lib"]

// MSAN-0-NOT: @__msan_track_origins
// MSAN-1:     @__msan_track_origins = weak_odr {{.*}}constant i32 1
// MSAN-2:     @__msan_track_origins = weak_odr {{.*}}constant i32 2
// MSAN-1-LTO: @__msan_track_origins = weak_odr {{.*}}constant i32 1
// MSAN-2-LTO: @__msan_track_origins = weak_odr {{.*}}constant i32 2
//
// MSAN-0-LABEL: define void @copy(
// MSAN-1-LABEL: define void @copy(
// MSAN-2-LABEL: define void @copy(
#[no_mangle]
pub fn copy(dst: &mut i32, src: &i32) {
    // MSAN-0-NOT: call i32 @__msan_chain_origin(
    // MSAN-1-NOT: call i32 @__msan_chain_origin(
    // MSAN-2:     call i32 @__msan_chain_origin(
    *dst = *src;
}
