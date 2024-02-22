//@ only-x86_64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type staticlib -Ctarget-feature=+sse4.2

// CHECK-LABEL: banana
// CHECK: crc32
#[no_mangle]
pub unsafe fn banana(v: u8) -> u32 {
    use std::arch::x86_64::*;
    let out = !0u32;
    _mm_crc32_u8(out, v)
}
