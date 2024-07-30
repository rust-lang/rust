//@ revisions: opt2 opt3winx64 opt3linx64
//@ [opt2] compile-flags: -O
//@ [opt3linx64] compile-flags: -C opt-level=3
//@ [opt3winx64] compile-flags: -C opt-level=3
//@ [opt3linx64] only-linux
//@ [opt3winx64] only-windows
//@ [opt3linx64] only-x86_64
//@ [opt3winx64] only-x86_64
//@ min-llvm-version: 18.1.3

#![crate_type = "lib"]
#![no_std]

// The code is from https://github.com/rust-lang/rust/issues/122805.
// Ensure we do not generate the shufflevector instruction
// to avoid complicating the code.
// CHECK-LABEL: define{{.*}}void @convert(
// CHECK-NOT: shufflevector
// CHECK-OPT2: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 2
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 4
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 6
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 8
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 10
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 12
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT2-NEXT: getelementptr inbounds i8, {{.+}} 14
// CHECK-OPT2-NEXT: store i16
// CHECK-OPT3LINX64: load <8 x i16>
// CHECK-OPT3LINX64-NEXT: call <8 x i16> @llvm.bswap
// CHECK-OPT3LINX64-NEXT: store <8 x i16>
// CHECK-OPT3WINX64: load <8 x i16>
// CHECK-OPT3WINX64-NEXT: call <8 x i16> @llvm.bswap
// CHECK-OPT3WINX64-NEXT: store <8 x i16>
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
