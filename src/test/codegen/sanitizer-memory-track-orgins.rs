// Verifies that MemorySanitizer track-origins level can be controlled
// with -Zsanitizer-memory-track-origins option.
//
// needs-sanitizer-support
// only-linux
// only-x86_64
// revisions:MSAN-0 MSAN-1 MSAN-2
//
//[MSAN-0] compile-flags: -Zsanitizer=memory
//[MSAN-1] compile-flags: -Zsanitizer=memory -Zsanitizer-memory-track-origins=1
//[MSAN-2] compile-flags: -Zsanitizer=memory -Zsanitizer-memory-track-origins

#![crate_type="lib"]

// MSAN-0-NOT: @__msan_track_origins
// MSAN-1:     @__msan_track_origins = weak_odr local_unnamed_addr constant i32 1
// MSAN-2:     @__msan_track_origins = weak_odr local_unnamed_addr constant i32 2
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
