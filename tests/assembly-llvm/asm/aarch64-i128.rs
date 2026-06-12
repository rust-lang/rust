//@ add-minicore
//@ revisions: aarch64 aarch64_be arm64ec
//@ assembly-output: emit-asm
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ [aarch64_be] compile-flags: --target aarch64_be-unknown-linux-gnu
//@ [aarch64_be] needs-llvm-components: aarch64
//@ [arm64ec] compile-flags: --target arm64ec-pc-windows-msvc
//@ [arm64ec] needs-llvm-components: aarch64
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, repr_simd, asm_experimental_reg)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

// Check how a 128-bit integer is passed to assembly. Note that on aarch64_be for i128
// the two 64-bit chunks are endian-swapped, while a SIMD type is passed as-is.

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

#[repr(simd)]
pub struct i8x16([i8; 16]);

impl Copy for i8x16 {}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal $modifier:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(
                concat!($mov, " {:", $modifier, "}, {:", $modifier, "}"),
                out($class) y,
                in($class) x
            );
            y
        }
    };
}

// CHECK-LABEL: {{("#)?}}vreg_i128{{"?}}
// CHECK: fmov d0, x0
// CHECK: mov v0.d[1], x1
// aarch64_be: rev64 v0.16b, v0.16b
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64_be: rev64 v0.16b, v1.16b
// CHECK: mov x1, v{{[0-9]+}}.d[1]
// CHECK: fmov x0, d{{[0-9]+}}
check!(vreg_i128 i128 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i8x16{{"?}}
// aarch64: ldr q0, [x0]
// aarch64_be: ld1 { v0.16b }, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str q1, [x8]
// aarch64_be: st1 { v1.16b }, [x8]
check!(vreg_i8x16 i8x16 vreg "fmov" "s");
