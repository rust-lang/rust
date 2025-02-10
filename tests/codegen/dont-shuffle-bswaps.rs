//@ revisions: OPT2 OPT3WINX64 OPT3LINX64
//@ [OPT2] compile-flags: -O
//@ [OPT3LINX64] compile-flags: -C opt-level=3
//@ [OPT3WINX64] compile-flags: -C opt-level=3
//@ [OPT3LINX64] only-linux
//@ [OPT3WINX64] only-windows
//@ [OPT3LINX64] only-x86_64
//@ [OPT3WINX64] only-x86_64
//@ min-llvm-version: 18.1.3

#![crate_type = "lib"]
#![no_std]

// The code is from https://github.com/rust-lang/rust/issues/122805.
// Ensure we do not generate the shufflevector instruction
// to avoid complicating the code.
// CHECK-LABEL: define{{.*}}void @convert(
// CHECK-NOT: shufflevector
// OPT2: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 2
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 4
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 6
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 8
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 10
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 12
// OPT2-NEXT: store i16
// OPT2-NEXT: getelementptr inbounds{{( nuw)?}} i8, {{.+}} 14
// OPT2-NEXT: store i16
// OPT3LINX64: load <8 x i16>
// OPT3LINX64-NEXT: call <8 x i16> @llvm.bswap
// OPT3LINX64-NEXT: store <8 x i16>
// OPT3WINX64: load <8 x i16>
// OPT3WINX64-NEXT: call <8 x i16> @llvm.bswap
// OPT3WINX64-NEXT: store <8 x i16>
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
