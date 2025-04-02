//@ revisions: OPT2 OPT3
//@[OPT2] compile-flags: -Copt-level=2
//@[OPT3] compile-flags: -C opt-level=3
// some targets don't do the opt we are looking for
//@[OPT3] only-64bit

#![crate_type = "lib"]
#![no_std]

// The code is from https://github.com/rust-lang/rust/issues/122805.
// Ensure we do not generate the shufflevector instruction
// to avoid complicating the code.
// CHECK-LABEL: define{{.*}}void @convert(
// CHECK-NOT: shufflevector
// On higher opt levels, this should just be a bswap:
// OPT3: load <8 x i16>
// OPT3-NEXT: call <8 x i16> @llvm.bswap
// OPT3-NEXT: store <8 x i16>
// OPT3-NEXT: ret void
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
