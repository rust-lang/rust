//@ compile-flags: -O
//@ min-llvm-version: 18.1.3

#![crate_type = "lib"]
#![no_std]

// The code is from https://github.com/rust-lang/rust/issues/122805.
// Ensure we do not generate the shufflevector instruction
// to avoid complicating the code.
// CHECK-LABEL: define{{.*}}void @convert(
// CHECK-NOT: shufflevector
// CHECK: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: insertelement <8 x i16>
// CHECK-NEXT: store <8 x i16>
// CHECK-NEXT: ret void
#[no_mangle]
#[cfg(target_endian = "little")]
pub fn convert(value: [u16; 8]) -> [u8; 16] {
    let addr16 = [
        value[0].to_be(),
        value[1].to_be(),
        value[2].to_be(),
        value[3].to_be(),
        value[4].to_be(),
        value[5].to_be(),
        value[6].to_be(),
        value[7].to_be(),
    ];
    unsafe { core::mem::transmute::<_, [u8; 16]>(addr16) }
}
