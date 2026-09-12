//@ add-minicore
//@ revisions: gfx942 gfx950 gfx1030
//@ assembly-output: emit-asm
//@ compile-flags: --target amdgcn-amd-amdhsa
//@[gfx942] compile-flags: -Ctarget-cpu=gfx942
//@[gfx950] compile-flags: -Ctarget-cpu=gfx950
//@[gfx1030] compile-flags: -Ctarget-cpu=gfx1030
//@ needs-llvm-components: amdgpu
//@ needs-rust-lld
// ignore-tidy-linelength

// Tests for different gfx versions that do not fit in gfx11 and 12

#![feature(abi_gpu_kernel, no_core, asm_experimental_arch, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(
    asm_sub_register,
    improper_gpu_kernel_arg,
    improper_ctypes_definitions,
    non_camel_case_types,
    unused_assignments,
    unused_variables
)]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

type ptr = *mut u8;

type i32x9 = Simd<i32, 9>;
type f32x9 = Simd<f32, 9>;

type i32x11 = Simd<i32, 11>;
type f32x11 = Simd<f32, 11>;

type i32x12 = Simd<i32, 12>;
type f32x12 = Simd<f32, 12>;

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func(x: $ty) {
            let y: $ty;
            asm!(concat!($mov, " {}, {}"), out($class) y, in($class) x);
        }
    };

    ($func:ident $ret_ty:ident $ret_class:ident $($arg_name:ident: $arg_ty:ident $arg_class:ident,)*
        $mov:literal) => {
        check!($func $ret_ty $ret_class $($arg_name: $arg_ty $arg_class,)* $mov, "");
    };

    ($func:ident $ret_ty:ident $ret_class:ident $($arg_name:ident: $arg_ty:ident $arg_class:ident,)*
        $mov:literal, $tail:literal) => {
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func($($arg_name: $arg_ty,)*) {
            let result: $ret_ty;
            asm!(concat!($mov, " {}", $(", {", stringify!($arg_name), "}",)* $tail),
                out($ret_class) result, $($arg_name = in($arg_class) $arg_name,)*);
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func(x: $ty) {
            let y: $ty;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        }
    };

    ($func:ident $ret_ty:ident $ret_reg:tt $($arg_name:ident: $arg_ty:ident $arg_reg:tt,)*
        $mov:literal) => {
        check_reg!($func $ret_ty $ret_reg $($arg_name: $arg_ty $arg_reg,)* $mov, "");
    };

    ($func:ident $ret_ty:ident $ret_reg:tt $($arg_name:ident: $arg_ty:ident $arg_reg:tt,)*
        $mov:literal, $tail:literal) => {
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func($($arg_name: $arg_ty,)*) {
            let result: $ret_ty;
            asm!(concat!($mov, " ", $ret_reg, $(", ", $arg_reg,)* $tail), lateout($ret_reg) result,
                $(in($arg_reg) $arg_name,)*);
        }
    };
}

// gfx1030-LABEL: vgpr_i32x9:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} a16
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_i32x9 i32x4 vgpr128 x: i32x9 vgpr288, y: i32x4 sgpr128, "image_bvh64_intersect_ray",
    " a16");

// gfx1030-LABEL: vgpr_f32x9:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} a16
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_f32x9 i32x4 vgpr128 x: f32x9 vgpr288, y: i32x4 sgpr128, "image_bvh64_intersect_ray",
    " a16");

// gfx1030-LABEL: vgpr_i32x11:
// gfx1030: #ASMSTART
// gfx1030: image_bvh_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_i32x11 i32x4 vgpr128 x: i32x11 vgpr352, y: i32x4 sgpr128, "image_bvh_intersect_ray");

// gfx1030-LABEL: vgpr_f32x11:
// gfx1030: #ASMSTART
// gfx1030: image_bvh_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_f32x11 i32x4 vgpr128 x: f32x11 vgpr352, y: i32x4 sgpr128, "image_bvh_intersect_ray");

// gfx1030-LABEL: vgpr_i32x12:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_i32x12 i32x4 vgpr128 x: i32x12 vgpr384, y: i32x4 sgpr128, "image_bvh64_intersect_ray");

// gfx1030-LABEL: vgpr_f32x12:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check!(vgpr_f32x12 i32x4 vgpr128 x: f32x12 vgpr384, y: i32x4 sgpr128, "image_bvh64_intersect_ray");

// gfx950-LABEL: vgpr_i32x16:
// gfx950: #ASMSTART
// gfx950: v_mfma_i32_32x32x32_i8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx950: #ASMEND
#[cfg(gfx950)]
check!(vgpr_i32x16 i32x16 vgpr512 x: i32x4 vgpr128, y: i32x4 vgpr128, z: i16x32 vgpr512,
    "v_mfma_i32_32x32x32_i8");

// gfx950-LABEL: vgpr_f32x16:
// gfx950: #ASMSTART
// gfx950: v_mfma_f32_32x32x16_f16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx950: #ASMEND
#[cfg(gfx950)]
check!(vgpr_f32x16 f32x16 vgpr512 x: f32x4 vgpr128, y: f32x4 vgpr128, z: f16x32 vgpr512,
    "v_mfma_f32_32x32x16_f16");

// gfx942-LABEL: vgpr_f32x32:
// gfx942: #ASMSTART
// gfx942: v_mfma_f32_32x32x1_2b_f32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx942: #ASMEND
#[cfg(gfx942)]
check!(vgpr_f32x32 f32x32 vgpr1024 x: f32 vgpr32, y: f32 vgpr32, "v_mfma_f32_32x32x1_2b_f32",
    ", v[0:31]");

// gfx1030-LABEL: v0_i32x9:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} a16
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_i32x9 i32x4 "v[0:3]" x: i32x9 "v[8:16]", y: i32x4 "s[0:3]",
    "image_bvh64_intersect_ray", " a16");

// gfx1030-LABEL: v0_f32x9:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} a16
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_f32x9 i32x4 "v[0:3]" x: f32x9 "v[8:16]", y: i32x4 "s[0:3]",
    "image_bvh64_intersect_ray", " a16");

// gfx1030-LABEL: v0_i32x11:
// gfx1030: #ASMSTART
// gfx1030: image_bvh_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_i32x11 i32x4 "v[0:3]" x: i32x11 "v[8:18]", y: i32x4 "s[0:3]",
    "image_bvh_intersect_ray");

// gfx1030-LABEL: v0_f32x11:
// gfx1030: #ASMSTART
// gfx1030: image_bvh_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_f32x11 i32x4 "v[0:3]" x: f32x11 "v[8:18]", y: i32x4 "s[0:3]",
    "image_bvh_intersect_ray");

// gfx1030-LABEL: v0_i32x12:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_i32x12 i32x4 "v[0:3]" x: i32x12 "v[8:19]", y: i32x4 "s[0:3]",
    "image_bvh64_intersect_ray");

// gfx1030-LABEL: v0_f32x12:
// gfx1030: #ASMSTART
// gfx1030: image_bvh64_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// gfx1030: #ASMEND
#[cfg(gfx1030)]
check_reg!(v0_f32x12 i32x4 "v[0:3]" x: f32x12 "v[8:19]", y: i32x4 "s[0:3]",
    "image_bvh64_intersect_ray");

// gfx950-LABEL: v0_i32x16:
// gfx950: #ASMSTART
// gfx950: v_mfma_i32_32x32x32_i8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx950: #ASMEND
#[cfg(gfx950)]
check_reg!(v0_i32x16 i32x16 "v[0:15]" x: i32x4 "v[16:19]", y: i32x4 "v[20:23]", z: i16x32 "v[0:15]",
    "v_mfma_i32_32x32x32_i8");

// gfx950-LABEL: v0_f32x16:
// gfx950: #ASMSTART
// gfx950: v_mfma_f32_32x32x16_f16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx950: #ASMEND
#[cfg(gfx950)]
check_reg!(v0_f32x16 f32x16 "v[0:15]" x: f32x4 "v[16:19]", y: f32x4 "v[20:23]", z: f16x32 "v[0:15]",
    "v_mfma_f32_32x32x16_f16");

// gfx942-LABEL: v0_f32x32:
// gfx942: #ASMSTART
// gfx942: v_mfma_f32_32x32x1_2b_f32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx942: #ASMEND
#[cfg(gfx942)]
check_reg!(v0_f32x32 f32x32 "v[0:31]" x: f32 "v32", y: f32 "v33", "v_mfma_f32_32x32x1_2b_f32",
    ", v[0:31]");
