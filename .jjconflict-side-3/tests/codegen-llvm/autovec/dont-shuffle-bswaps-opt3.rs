//@ revisions: AARCH64 X86_64 Z13
//@ compile-flags: -Copt-level=3
//@[AARCH64] only-aarch64
//@[X86_64] only-x86_64
//@[Z13] only-s390x
//@[Z13] compile-flags: -Ctarget-cpu=z13

#![crate_type = "lib"]
#![no_std]

// This test is paired with the arch-neutral -opt2.rs test

// The code is from https://github.com/rust-lang/rust/issues/122805.
// Ensure we do not generate the shufflevector instruction
// to avoid complicating the code.

// CHECK-LABEL: define{{.*}}void @convert(
// CHECK-NOT: shufflevector

// On higher opt levels, this should just be a bswap:
// CHECK: load <8 x i16>
// CHECK-NEXT: call <8 x i16> @llvm.bswap
// CHECK-NEXT: store <8 x i16>
// CHECK-NEXT: ret void
#[no_mangle]
pub fn convert(value: [u16; 8]) -> [u8; 16] {
    #[cfg(target_endian = "little")]
    let bswap = u16::to_be;
    #[cfg(target_endian = "big")]
    let bswap = u16::to_le;
    let addr16 = [
        bswap(value[0]),
        bswap(value[1]),
        bswap(value[2]),
        bswap(value[3]),
        bswap(value[4]),
        bswap(value[5]),
        bswap(value[6]),
        bswap(value[7]),
    ];
    unsafe { core::mem::transmute::<_, [u8; 16]>(addr16) }
}
