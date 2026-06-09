//@ assembly-output: emit-asm
// # zen3 previously exhibited odd vectorization
//@ compile-flags: --crate-type=lib -Ctarget-cpu=znver3 -Copt-level=3
//@ only-x86_64
//@ ignore-sgx

use std::iter;

// previously this produced a long chain of
// 56:  vpextrb $6, %xmm0, %ecx
// 57:  orb %cl, 22(%rsi)
// 58:  vpextrb $7, %xmm0, %ecx
// 59:  orb %cl, 23(%rsi)
// [...]

// CHECK-LABEL: zip_arrays:
#[no_mangle]
pub fn zip_arrays(mut a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
    // CHECK-NOT: vpextrb
    // CHECK-NOT: orb %cl
    // CHECK: vorps
    iter::zip(&mut a, b).for_each(|(a, b)| *a |= b);
    // CHECK: retq
    a
}
