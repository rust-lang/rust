// ignore-tidy-file-linelength (some revision //@ lines are over 100 chars long)

//@ add-minicore
//@ revisions: mips32 mips32el mips32r6 mips32r6el mips64 mips64el mips64r6 mips64r6el
//@ revisions: mips32_msa mips32el_msa mips32r6_msa mips32r6el_msa mips64_msa mips64el_msa mips64r6_msa mips64r6el_msa
//@ assembly-output: emit-asm

//@[mips32] compile-flags: --target mips-unknown-linux-gnu
//@[mips32] needs-llvm-components: mips
//@[mips32el] compile-flags: --target mipsel-unknown-linux-gnu --cfg mips32
//@[mips32el] needs-llvm-components: mips
//@[mips32el] filecheck-flags: --check-prefix mips32

//@[mips32r6] compile-flags: --target mipsisa32r6-unknown-linux-gnu --cfg mips32
//@[mips32r6] needs-llvm-components: mips
//@[mips32r6] filecheck-flags: --check-prefix mips32
//@[mips32r6el] compile-flags: --target mipsisa32r6el-unknown-linux-gnu --cfg mips32
//@[mips32r6el] needs-llvm-components: mips
//@[mips32r6el] filecheck-flags: --check-prefix mips32

//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64 --cfg mips64_not_r6
//@[mips64] needs-llvm-components: mips
//@[mips64] filecheck-flags: --check-prefix mips64-not-r6
//@[mips64el] compile-flags: --target mips64el-unknown-linux-gnuabi64 --cfg mips64 --cfg mips64_not_r6
//@[mips64el] needs-llvm-components: mips
//@[mips64el] filecheck-flags: --check-prefix mips64 --check-prefix not-r6

//@[mips64r6] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64 --cfg mips64
//@[mips64r6] needs-llvm-components: mips
//@[mips64r6] filecheck-flags: --check-prefix mips64
//@[mips64r6el] compile-flags: --target mipsisa64r6el-unknown-linux-gnuabi64 --cfg mips64 --cfg mips64r6
//@[mips64r6el] needs-llvm-components: mips
//@[mips64r6el] filecheck-flags: --check-prefix mips64 --check-prefix mips64r6

//@[mips32_msa] compile-flags: --target mips-unknown-linux-gnu -Ctarget-feature=+fp64,+mips32r5,+msa --cfg mips32 --cfg msa
//@[mips32_msa] needs-llvm-components: mips
//@[mips32_msa] filecheck-flags: --check-prefix mips32 --check-prefix msa
//@[mips32el_msa] compile-flags: --target mipsel-unknown-linux-gnu -Ctarget-feature=+fp64,+mips32r5,+msa --cfg mips32 --cfg msa
//@[mips32el_msa] needs-llvm-components: mips
//@[mips32el_msa] filecheck-flags: --check-prefix mips32 --check-prefix msa

//@[mips32r6_msa] compile-flags: --target mipsisa32r6-unknown-linux-gnu -Ctarget-feature=+msa --cfg mips32 --cfg msa
//@[mips32r6_msa] needs-llvm-components: mips
//@[mips32r6_msa] filecheck-flags: --check-prefix mips32 --check-prefix msa
//@[mips32r6el_msa] compile-flags: --target mipsisa32r6el-unknown-linux-gnu -Ctarget-feature=+msa --cfg mips32 --cfg msa
//@[mips32r6el_msa] needs-llvm-components: mips
//@[mips32r6el_msa] filecheck-flags: --check-prefix mips32 --check-prefix msa

//@[mips64_msa] compile-flags: --target mips64-unknown-linux-gnuabi64 -Ctarget-feature=+mips64r5,+msa --cfg mips64 --cfg mips64_not_r6 --cfg msa
//@[mips64_msa] needs-llvm-components: mips
//@[mips64_msa] filecheck-flags: --check-prefix mips64 --check-prefix mips64-not-r6 --check-prefix msa
//@[mips64el_msa] compile-flags: --target mips64el-unknown-linux-gnuabi64 -Ctarget-feature=+mips64r5,+msa --cfg mips64 --cfg mips64_not_r6 --cfg msa
//@[mips64el_msa] needs-llvm-components: mips
//@[mips64el_msa] filecheck-flags: --check-prefix mips64 --check-prefix not-r6 --check-prefix msa

//@[mips64r6_msa] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64 -Ctarget-feature=+msa --cfg mips64 --cfg mips64r6 --cfg msa
//@[mips64r6_msa] needs-llvm-components: mips
//@[mips64r6_msa] filecheck-flags: --check-prefix mips64 --check-prefix mips64r6 --check-prefix msa
//@[mips64r6el_msa] compile-flags: --target mipsisa64r6el-unknown-linux-gnuabi64 -Ctarget-feature=+msa --cfg mips64 --cfg mips64r6 --cfg msa
//@[mips64r6el_msa] needs-llvm-components: mips
//@[mips64r6el_msa] filecheck-flags: --check-prefix mips64 --check-prefix mips64r6 --check-prefix msa

//@ compile-flags: -Zmerge-functions=disabled
//@ compile-flags: --check-cfg=cfg(mips64_not_r6,msa)
// `f16` causes LLVM to crash when `msa` is enabled on LLVM < 23
//@ min-llvm-version: 23

#![deny(unexpected_cfgs)]
#![feature(no_core, asm_experimental_arch, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types, unused)]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

#[cfg_attr(mips32, cfg(not(target_pointer_width = "32")))]
#[cfg_attr(not(mips32), cfg(target_pointer_width = "32"))]
compile_error!("mips32 cfg mismatch");

#[cfg_attr(mips64, cfg(not(target_pointer_width = "64")))]
#[cfg_attr(not(mips64), cfg(target_pointer_width = "64"))]
compile_error!("mips64 cfg mismatch");

#[cfg_attr(mips64_not_r6, cfg(not(target_arch = "mips64")))]
#[cfg_attr(not(mips64_not_r6), cfg(target_arch = "mips64"))]
compile_error!("mips64_not_r6 cfg mismatch");

#[cfg_attr(mips64r6, cfg(not(target_arch = "mips64r6")))]
#[cfg_attr(not(mips64r6), cfg(target_arch = "mips64r6"))]
compile_error!("mips64r6 cfg mismatch");

#[cfg_attr(msa, cfg(not(target_feature = "msa")))]
#[cfg_attr(not(msa), cfg(target_feature = "msa"))]
compile_error!("msa cfg mismatch");

type ptr = *const i32;

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[unsafe(no_mangle)]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    #[unsafe(no_mangle)]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// mips32-LABEL: sym_static_32:
// mips32: #APP
// mips32: lw $3, %got(extern_static)($gp)
// mips32: #NO_APP
#[cfg(mips32)]
#[no_mangle]
pub unsafe fn sym_static_32() {
    asm!("lw $v1, {}", sym extern_static);
}

// mips32-LABEL: sym_fn_32:
// mips32: #APP
// mips32: lw $3, %got(extern_func)($gp)
// mips32: #NO_APP
#[cfg(mips32)]
#[no_mangle]
pub unsafe fn sym_fn_32() {
    asm!("lw $v1, {}", sym extern_func);
}

// mips64-LABEL: sym_static_64:
// mips64: #APP
// mips64-not-r6: lui    $3, %got_hi(extern_static)
// mips64-not-r6: daddu  $3, $3, $gp
// mips64-not-r6: ld     $3, %got_lo(extern_static)($3)
// mips64r6: ld $3, %got_disp(extern_static)($gp)
// mips64: #NO_APP
#[cfg(mips64)]
#[no_mangle]
pub unsafe fn sym_static_64() {
    asm!("ld $v1, {}", sym extern_static);
}

// mips64-LABEL: sym_fn_64:
// mips64: #APP
// mips64-not-r6: lui    $3, %got_hi(extern_func)
// mips64-not-r6: daddu  $3, $3, $gp
// mips64-not-r6: ld     $3, %got_lo(extern_func)($3)
// mips64r6: ld $3, %got_disp(extern_func)($gp)
// mips64: #NO_APP
#[cfg(mips64)]
#[no_mangle]
pub unsafe fn sym_fn_64() {
    asm!("ld $v1, {}", sym extern_func);
}

// CHECK-LABEL: reg_f16:
// CHECK: #APP
// CHECK: mov.s $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f16, f16, freg, "mov.s");

// CHECK-LABEL: f0_f16:
// CHECK: #APP
// CHECK: mov.s $f0, $f0
// CHECK: #NO_APP
check_reg!(f0_f16, f16, "$f0", "mov.s");

// CHECK-LABEL: reg_f16_64:
// CHECK: #APP
// CHECK: mov.d $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f16_64, f16, freg, "mov.d");

// CHECK-LABEL: f0_f16_64:
// CHECK: #APP
// CHECK: mov.d $f0, $f0
// CHECK: #NO_APP
check_reg!(f0_f16_64, f16, "$f0", "mov.d");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: mov.s $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "mov.s");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: mov.s $f0, $f0
// CHECK: #NO_APP
check_reg!(f0_f32, f32, "$f0", "mov.s");

// CHECK-LABEL: reg_f32_64:
// CHECK: #APP
// CHECK: mov.d $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32_64, f32, freg, "mov.d");

// CHECK-LABEL: f0_f32_64:
// CHECK: #APP
// CHECK: mov.d $f0, $f0
// CHECK: #NO_APP
check_reg!(f0_f32_64, f32, "$f0", "mov.d");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: mov.d $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f64, f64, freg, "mov.d");

// CHECK-LABEL: f0_f64:
// CHECK: #APP
// CHECK: mov.d $f0, $f0
// CHECK: #NO_APP
check_reg!(f0_f64, f64, "$f0", "mov.d");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg, "move");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "move");

// CHECK-LABEL: reg_f16_soft:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f16_soft, f16, reg, "move");

// CHECK-LABEL: reg_f32_soft:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32_soft, f32, reg, "move");

// mips64-LABEL: reg_f64_soft:
// mips64: #APP
// mips64: move ${{[0-9]+}}, ${{[0-9]+}}
// mips64: #NO_APP
#[cfg(mips64)]
check!(reg_f64_soft, f64, reg, "move");

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "move");

// CHECK-LABEL: reg_u8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_u8, u8, reg, "move");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "move");

// mips64-LABEL: reg_i64:
// mips64: #APP
// mips64: move ${{[0-9]+}}, ${{[0-9]+}}
// mips64: #NO_APP
#[cfg(mips64)]
check!(reg_i64, i64, reg, "move");

// CHECK-LABEL: r8_ptr:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_ptr, ptr, "$8", "move");

// CHECK-LABEL: r8_i32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i32, i32, "$8", "move");

// CHECK-LABEL: r8_f16:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_f16, f16, "$8", "move");

// CHECK-LABEL: r8_f32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_f32, f32, "$8", "move");

// CHECK-LABEL: r8_i8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i8, i8, "$8", "move");

// CHECK-LABEL: r8_u8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_u8, u8, "$8", "move");

// CHECK-LABEL: r8_i16:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i16, i16, "$8", "move");

// msa-LABEL: wreg_f16:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f16, f16, wreg, "move.v");

// msa-LABEL: wreg_f32:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f32, f32, wreg, "move.v");

// msa-LABEL: wreg_f64:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f64, f64, wreg, "move.v");

// msa-LABEL: w0_f16:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f16, f16, "$w0", "move.v");

// msa-LABEL: w0_f32:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f32, f32, "$w0", "move.v");

// msa-LABEL: w0_f64:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f64, f64, "$w0", "move.v");

// msa-LABEL: wreg_i8x16:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_i8x16, i8x16, wreg, "move.v");

// msa-LABEL: wreg_i16x8:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_i16x8, i16x8, wreg, "move.v");

// msa-LABEL: wreg_i32x4:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_i32x4, i32x4, wreg, "move.v");

// msa-LABEL: wreg_i64x2:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_i64x2, i64x2, wreg, "move.v");

// msa-LABEL: wreg_f16x8:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f16x8, f16x8, wreg, "move.v");

// msa-LABEL: wreg_f32x4:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f32x4, f32x4, wreg, "move.v");

// msa-LABEL: wreg_f64x2:
// msa: #APP
// msa: move.v $w{{[0-9]+}}, $w{{[0-9]+}}
// msa: #NO_APP
#[cfg(msa)]
check!(wreg_f64x2, f64x2, wreg, "move.v");

// msa-LABEL: w0_i8x16:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_i8x16, i8x16, "$w0", "move.v");

// msa-LABEL: w0_i16x8:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_i16x8, i16x8, "$w0", "move.v");

// msa-LABEL: w0_i32x4:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_i32x4, i32x4, "$w0", "move.v");

// msa-LABEL: w0_i64x2:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_i64x2, i64x2, "$w0", "move.v");

// msa-LABEL: w0_f16x8:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f16x8, f16x8, "$w0", "move.v");

// msa-LABEL: w0_f32x4:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f32x4, f32x4, "$w0", "move.v");

// msa-LABEL: w0_f64x2:
// msa: #APP
// msa: move.v $w0, $w0
// msa: #NO_APP
#[cfg(msa)]
check_reg!(w0_f64x2, f64x2, "$w0", "move.v");
