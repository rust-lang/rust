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

#![feature(no_core, f128, asm_experimental_reg)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

// Check how a 128-bit integer is passed to assembly. Note that on aarch64_be for i128
// the two 64-bit chunks are endian-swapped, while a SIMD type is passed as-is.

extern crate minicore;
use minicore::simd::*;
use minicore::*;

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
// FIXME(llvm23) arm64ec only supports f128 from LLVM23 onwards.
//
// aarch64_be-LABEL: {{("#)?}}vreg_f128{{"?}}
// aarch64_be: rev64 v0.16b, v0.16b
// aarch64_be: //APP
// aarch64_be: fmov s{{[0-9]+}}, s{{[0-9]+}}
// aarch64_be: //NO_APP
// aarch64_be: rev64 v0.16b, v1.16b
// aarch64_be: ret
//
// aarch64-LABEL: {{("#)?}}vreg_f128{{"?}}
// aarch64: //APP
// aarch64: fmov s{{[0-9]+}}, s{{[0-9]+}}
// aarch64: //NO_APP
// aarch64: ret
#[cfg(target_arch = "aarch64")]
check!(vreg_f128 f128 vreg "fmov" "s");

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

// CHECK-LABEL: {{("#)?}}vreg_i16x8{{"?}}
// aarch64: ldr q0, [x0]
// aarch64_be: ld1 { v0.16b }, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str q1, [x8]
// aarch64_be: st1 { v1.16b }, [x8]
check!(vreg_i16x8 i16x8 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i32x4{{"?}}
// aarch64: ldr q0, [x0]
// aarch64_be: ld1 { v0.16b }, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str q1, [x8]
// aarch64_be: st1 { v1.16b }, [x8]
check!(vreg_i32x4 i32x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i64x2{{"?}}
// aarch64: ldr q0, [x0]
// aarch64_be: ld1 { v0.16b }, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str q1, [x8]
// aarch64_be: st1 { v1.16b }, [x8]
check!(vreg_i64x2 i64x2 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i16x4{{"?}}
// aarch64: ldr d0, [x0]
// aarch64_be: ldr d0, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str d1, [x8]
// aarch64_be: str d1, [x8]
check!(vreg_i16x4 i16x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i32x2{{"?}}
// aarch64: ldr d0, [x0]
// aarch64_be: ldr d0, [x0]
// aarch64_be-NOT: rev64
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
// aarch64: str d1, [x8]
// aarch64_be: str d1, [x8]
check!(vreg_i32x2 i32x2 vreg "fmov" "s");
