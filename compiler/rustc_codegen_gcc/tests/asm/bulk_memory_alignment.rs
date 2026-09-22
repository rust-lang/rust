//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// The alignment reaches GCC's `memcpy`/`memset` expansion only through
// `__builtin_assume_aligned`; a pointer cast to an aligned type is stripped as a useless
// conversion. An over-aligned type therefore has to expand to aligned moves and a packed one
// to unaligned moves. The alignment is 64 so that the contrast holds whatever vector width
// the host picks.

#[repr(align(64))]
pub struct Aligned([u8; 64]);

#[repr(C, packed)]
pub struct Packed([u8; 64]);

// CHECK-LABEL: "copy_aligned":
// CHECK: {{(v)?mov(dqa|aps)}}
#[no_mangle]
pub unsafe fn copy_aligned(destination: *mut Aligned, source: *const Aligned) {
    core::ptr::copy_nonoverlapping(source, destination, 1);
}

// CHECK-LABEL: "copy_packed":
// CHECK: {{(v)?mov(dqu|ups)}}
#[no_mangle]
pub unsafe fn copy_packed(destination: *mut Packed, source: *const Packed) {
    core::ptr::copy_nonoverlapping(source, destination, 1);
}

// CHECK-LABEL: "set_aligned":
// CHECK: {{(v)?mov(dqa|aps)}}
#[no_mangle]
pub unsafe fn set_aligned(destination: *mut Aligned) {
    core::ptr::write_bytes(destination, 0, 1);
}

// CHECK-LABEL: "set_packed":
// CHECK: {{(v)?mov(dqu|ups)}}
#[no_mangle]
pub unsafe fn set_packed(destination: *mut Packed) {
    core::ptr::write_bytes(destination, 0, 1);
}
