//@ add-minicore
//@ revisions: gfx11 gfx12
//@ assembly-output: emit-asm
//@ compile-flags: --target amdgcn-amd-amdhsa
//@[gfx11] compile-flags: -Ctarget-cpu=gfx1100
//@[gfx12] compile-flags: -Ctarget-cpu=gfx1200
//@ needs-llvm-components: amdgpu
//@ needs-rust-lld
// ignore-tidy-linelength

#![feature(abi_gpu_kernel, no_core, asm_experimental_arch, repr_simd, f16)]
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
use minicore::*;

type ptr = *mut u8;

#[repr(simd)]
pub struct i16x2([i16; 2]);
#[repr(simd)]
pub struct f16x2([f16; 2]);

#[repr(simd)]
pub struct i16x4([i16; 4]);
#[repr(simd)]
pub struct f16x4([f16; 4]);
#[repr(simd)]
pub struct i32x2([i32; 2]);
#[repr(simd)]
pub struct f32x2([f32; 2]);

#[repr(simd)]
pub struct i32x3([i32; 3]);
#[repr(simd)]
pub struct f32x3([f32; 3]);

#[repr(simd)]
pub struct i16x8([i16; 8]);
#[repr(simd)]
pub struct f16x8([f16; 8]);
#[repr(simd)]
pub struct i32x4([i32; 4]);
#[repr(simd)]
pub struct f32x4([f32; 4]);
#[repr(simd)]
pub struct i64x2([i64; 2]);
#[repr(simd)]
pub struct f64x2([f64; 2]);

#[repr(simd)]
pub struct i32x5([i32; 5]);
#[repr(simd)]
pub struct f32x5([f32; 5]);

#[repr(simd)]
pub struct i32x6([i32; 6]);
#[repr(simd)]
pub struct f32x6([f32; 6]);
#[repr(simd)]
pub struct i64x3([i64; 3]);
#[repr(simd)]
pub struct f64x3([f64; 3]);

#[repr(simd)]
pub struct i32x7([i32; 7]);
#[repr(simd)]
pub struct f32x7([f32; 7]);

#[repr(simd)]
pub struct i16x16([i16; 16]);
#[repr(simd)]
pub struct f16x16([f16; 16]);
#[repr(simd)]
pub struct i32x8([i32; 8]);
#[repr(simd)]
pub struct f32x8([f32; 8]);
#[repr(simd)]
pub struct i64x4([i64; 4]);
#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct i32x10([i32; 10]);
#[repr(simd)]
pub struct f32x10([f32; 10]);

#[repr(simd)]
pub struct i16x32([i16; 32]);
#[repr(simd)]
pub struct f16x32([f16; 32]);
#[repr(simd)]
pub struct i32x16([i32; 16]);
#[repr(simd)]
pub struct f32x16([f32; 16]);
#[repr(simd)]
pub struct i64x8([i64; 8]);
#[repr(simd)]
pub struct f64x8([f64; 8]);

macro_rules! impl_copy {
    ($($ty:ident)*) => {
        $(
            impl Copy for $ty {}
        )*
    };
}

impl_copy!(
    i16x2 f16x2 i16x4 f16x4 i32x2 f32x2 i32x3 f32x3 i16x8 f16x8 i32x4 f32x4
    i64x2 f64x2 i32x5 f32x5 i32x6 f32x6 i64x3 f64x3 i32x7 f32x7 i16x16 f16x16
    i32x8 f32x8 i64x4 f64x4 i32x10 f32x10 i16x32 f16x32 i32x16 f32x16 i64x8
    f64x8
);

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

// CHECK-LABEL: sgpr_i16x2:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check!(sgpr_i16x2 i16x2 sgpr32 x: i16 sgpr32, y: i16 sgpr32, "s_pack_ll_b32_b16");

// CHECK-LABEL: sgpr_f16x2:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check!(sgpr_f16x2 f16x2 sgpr32 x: i16 sgpr32, y: i16 sgpr32, "s_pack_ll_b32_b16");

// CHECK-LABEL: vgpr_i16x2:
// CHECK: #ASMSTART
// CHECK: v_pk_add_i16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// CHECK: #ASMEND
check!(vgpr_i16x2 i16x2 vgpr32 x: i16x2 vgpr32, y: i16x2 vgpr32, "v_pk_add_i16");

// CHECK-LABEL: vgpr_f16x2:
// CHECK: #ASMSTART
// CHECK: v_pk_add_f16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// CHECK: #ASMEND
check!(vgpr_f16x2 f16x2 vgpr32 x: f16x2 vgpr32, y: f16x2 vgpr32, "v_pk_add_f16");

// CHECK-LABEL: sgpr_i16x4:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i16x4 i16x4 sgpr64 x: ptr sgpr64, y: i32 sgpr32, "s_load_b64");

// CHECK-LABEL: sgpr_f16x4:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f16x4 f16x4 sgpr64 x: ptr sgpr64, y: i32 sgpr32, "s_load_b64");

// CHECK-LABEL: sgpr_i32x2:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i32x2 i32x2 sgpr64 x: ptr sgpr64, y: i32 sgpr32, "s_load_b64");

// CHECK-LABEL: sgpr_f32x2:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f32x2 f32x2 sgpr64 x: ptr sgpr64, y: i32 sgpr32, "s_load_b64");

// CHECK-LABEL: vgpr_i16x4:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i16x4 i16x4 vgpr64 x: i32 vgpr32, y: ptr sgpr64, "global_load_b64");

// CHECK-LABEL: vgpr_f16x4:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f16x4 f16x4 vgpr64 x: i32 vgpr32, y: ptr sgpr64, "global_load_b64");

// CHECK-LABEL: vgpr_i32x2:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i32x2 i32x2 vgpr64 x: i32 vgpr32, y: ptr sgpr64, "global_load_b64");

// CHECK-LABEL: vgpr_f32x2:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f32x2 f32x2 vgpr64 x: i32 vgpr32, y: ptr sgpr64, "global_load_b64");

// gfx12-LABEL: sgpr_i32x3:
// gfx12: #ASMSTART
// gfx12: s_load_b96 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(sgpr_i32x3 i32x3 sgpr96 x: ptr sgpr64, y: i32 sgpr32, "s_load_b96");

// gfx12-LABEL: sgpr_f32x3:
// gfx12: #ASMSTART
// gfx12: s_load_b96 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(sgpr_f32x3 f32x3 sgpr96 x: ptr sgpr64, y: i32 sgpr32, "s_load_b96");

// CHECK-LABEL: vgpr_i32x3:
// CHECK: #ASMSTART
// CHECK: global_load_b96 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i32x3 i32x3 vgpr96 x: i32 vgpr32, y: ptr sgpr64, "global_load_b96");

// CHECK-LABEL: vgpr_f32x3:
// CHECK: #ASMSTART
// CHECK: global_load_b96 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f32x3 f32x3 vgpr96 x: i32 vgpr32, y: ptr sgpr64, "global_load_b96");

// CHECK-LABEL: sgpr_i16x8:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i16x8 i16x8 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: sgpr_f16x8:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f16x8 f16x8 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: sgpr_i32x4:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i32x4 i32x4 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: sgpr_f32x4:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f32x4 f32x4 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: sgpr_i64x2:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i64x2 i64x2 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: sgpr_f64x2:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f64x2 f64x2 sgpr128 x: ptr sgpr64, y: i32 sgpr32, "s_load_b128");

// CHECK-LABEL: vgpr_i16x8:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i16x8 i16x8 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_f16x8:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f16x8 f16x8 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_i32x4:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i32x4 i32x4 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_f32x4:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f32x4 f32x4 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_i64x2:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i64x2 i64x2 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_f64x2:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f64x2 f64x2 vgpr128 x: i32 vgpr32, y: ptr sgpr64, "global_load_b128");

// CHECK-LABEL: vgpr_i32x5:
// CHECK: #ASMSTART
// CHECK: image_load v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_1D tfe
// CHECK: #ASMEND
check!(vgpr_i32x5 i32x5 vgpr160 x: i32 vgpr32, y: i32x8 sgpr256, "image_load",
    " dmask:0xf dim:SQ_RSRC_IMG_1D tfe");

// CHECK-LABEL: vgpr_f32x5:
// CHECK: #ASMSTART
// CHECK: image_load v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_1D tfe
// CHECK: #ASMEND
check!(vgpr_f32x5 f32x5 vgpr160 x: i32 vgpr32, y: i32x8 sgpr256, "image_load",
    " dmask:0xf dim:SQ_RSRC_IMG_1D tfe");

// gfx11-LABEL: vgpr_i32x6:
// gfx11: #ASMSTART
// gfx11: image_sample_d v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_i32x6 i32x4 vgpr128 x: i32x6 vgpr192, y: i32x8 sgpr256, z: i32x4 sgpr128,
    "image_sample_d", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: vgpr_f32x6:
// gfx11: #ASMSTART
// gfx11: image_sample_d v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_f32x6 i32x4 vgpr128 x: f32x6 vgpr192, y: i32x8 sgpr256, z: i32x4 sgpr128,
    "image_sample_d", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: vgpr_i32x7:
// gfx11: #ASMSTART
// gfx11: image_sample_d_cl v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_i32x7 i32x4 vgpr128 x: i32x7 vgpr224, y: i32x8 sgpr256, z: i32x4 sgpr128,
    "image_sample_d_cl", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: vgpr_f32x7:
// gfx11: #ASMSTART
// gfx11: image_sample_d_cl v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_f32x7 i32x4 vgpr128 x: f32x7 vgpr224, y: i32x8 sgpr256, z: i32x4 sgpr128,
    "image_sample_d_cl", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// CHECK-LABEL: sgpr_i16x16:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i16x16 i16x16 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// CHECK-LABEL: sgpr_f16x16:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f16x16 f16x16 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// CHECK-LABEL: sgpr_i32x8:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i32x8 i32x8 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// CHECK-LABEL: sgpr_f32x8:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f32x8 f32x8 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// CHECK-LABEL: sgpr_i64x4:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i64x4 i64x4 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// CHECK-LABEL: sgpr_f64x4:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f64x4 f64x4 sgpr256 x: ptr sgpr64, y: i32 sgpr32, "s_load_b256");

// gfx11-LABEL: vgpr_i16x16:
// gfx11: #ASMSTART
// gfx11: v_wmma_f32_16x16x16_bf16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_i16x16 f32x8 vgpr256 x: i32x8 vgpr256, y: i16x16 vgpr256, z: f32x8 vgpr256,
    "v_wmma_f32_16x16x16_bf16");

// gfx11-LABEL: vgpr_f16x16:
// gfx11: #ASMSTART
// gfx11: v_wmma_f32_16x16x16_f16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_f16x16 f32x8 vgpr256 x: f16x16 vgpr256, y: f16x16 vgpr256, z: f32x8 vgpr256,
    "v_wmma_f32_16x16x16_f16");

// gfx11-LABEL: vgpr_i32x8:
// gfx11: #ASMSTART
// gfx11: v_wmma_i32_16x16x16_iu8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_i32x8 i32x8 vgpr256 x: i32x4 vgpr128, y: i32x4 vgpr128, z: i32x8 vgpr256,
    "v_wmma_i32_16x16x16_iu8");

// gfx12-LABEL: vgpr_f32x8:
// gfx12: #ASMSTART
// gfx12: v_wmma_f32_16x16x16_fp8_fp8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(vgpr_f32x8 f32x8 vgpr256 x: f32x2 vgpr64, y: f32x2 vgpr64, z: f32x8 vgpr256,
    "v_wmma_f32_16x16x16_fp8_fp8");

// gfx12-LABEL: vgpr_i32x10:
// gfx12: #ASMSTART
// gfx12: image_bvh8_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(vgpr_i32x10 i32x10 vgpr320 "image_bvh8_intersect_ray",
    ", [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]");

// gfx12-LABEL: vgpr_f32x10:
// gfx12: #ASMSTART
// gfx12: image_bvh8_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(vgpr_f32x10 f32x10 vgpr320 "image_bvh8_intersect_ray",
    ", [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]");

// CHECK-LABEL: sgpr_i16x32:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i16x32 i16x32 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: sgpr_f16x32:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f16x32 f16x32 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: sgpr_i32x16:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i32x16 i32x16 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: sgpr_f32x16:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f32x16 f32x16 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: sgpr_i64x8:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i64x8 i64x8 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: sgpr_f64x8:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f64x8 f64x8 sgpr512 x: ptr sgpr64, y: i32 sgpr32, "s_load_b512");

// CHECK-LABEL: s0_i16x2:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check_reg!(s0_i16x2 i16x2 "s0" x: i16 "s1", y: i16 "s2", "s_pack_ll_b32_b16");

// CHECK-LABEL: s0_f16x2:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check_reg!(s0_f16x2 f16x2 "s0" x: i16 "s1", y: i16 "s2", "s_pack_ll_b32_b16");

// CHECK-LABEL: v0_i16x2:
// CHECK: #ASMSTART
// CHECK: v_pk_add_i16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// CHECK: #ASMEND
check_reg!(v0_i16x2 i16x2 "v0" x: i16x2 "v1", y: i16x2 "v2", "v_pk_add_i16");

// CHECK-LABEL: v0_f16x2:
// CHECK: #ASMSTART
// CHECK: v_pk_add_f16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// CHECK: #ASMEND
check_reg!(v0_f16x2 f16x2 "v0" x: f16x2 "v1", y: f16x2 "v2", "v_pk_add_f16");

// CHECK-LABEL: s0_i16x4:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i16x4 i16x4 "s[0:1]" x: ptr "s[2:3]", y: i32 "s4", "s_load_b64");

// CHECK-LABEL: s0_f16x4:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f16x4 f16x4 "s[0:1]" x: ptr "s[2:3]", y: i32 "s4", "s_load_b64");

// CHECK-LABEL: s0_i32x2:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i32x2 i32x2 "s[0:1]" x: ptr "s[2:3]", y: i32 "s4", "s_load_b64");

// CHECK-LABEL: s0_f32x2:
// CHECK: #ASMSTART
// CHECK: s_load_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f32x2 f32x2 "s[0:1]" x: ptr "s[2:3]", y: i32 "s4", "s_load_b64");

// CHECK-LABEL: v0_i16x4:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i16x4 i16x4 "v[0:1]" x: i32 "v2", y: ptr "s[0:1]", "global_load_b64");

// CHECK-LABEL: v0_f16x4:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f16x4 f16x4 "v[0:1]" x: i32 "v2", y: ptr "s[0:1]", "global_load_b64");

// CHECK-LABEL: v0_i32x2:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i32x2 i32x2 "v[0:1]" x: i32 "v2", y: ptr "s[0:1]", "global_load_b64");

// CHECK-LABEL: v0_f32x2:
// CHECK: #ASMSTART
// CHECK: global_load_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f32x2 f32x2 "v[0:1]" x: i32 "v2", y: ptr "s[0:1]", "global_load_b64");

// gfx12-LABEL: s0_i32x3:
// gfx12: #ASMSTART
// gfx12: s_load_b96 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(s0_i32x3 i32x3 "s[0:2]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b96");

// gfx12-LABEL: s0_f32x3:
// gfx12: #ASMSTART
// gfx12: s_load_b96 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(s0_f32x3 f32x3 "s[0:2]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b96");

// CHECK-LABEL: v0_i32x3:
// CHECK: #ASMSTART
// CHECK: global_load_b96 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i32x3 i32x3 "v[0:2]" x: i32 "v3", y: ptr "s[0:1]", "global_load_b96");

// CHECK-LABEL: v0_f32x3:
// CHECK: #ASMSTART
// CHECK: global_load_b96 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f32x3 f32x3 "v[0:2]" x: i32 "v3", y: ptr "s[0:1]", "global_load_b96");

// CHECK-LABEL: s0_i16x8:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i16x8 i16x8 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: s0_f16x8:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f16x8 f16x8 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: s0_i32x4:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i32x4 i32x4 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: s0_f32x4:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f32x4 f32x4 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: s0_i64x2:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i64x2 i64x2 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: s0_f64x2:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f64x2 f64x2 "s[0:3]" x: ptr "s[4:5]", y: i32 "s6", "s_load_b128");

// CHECK-LABEL: v0_i16x8:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i16x8 i16x8 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_f16x8:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f16x8 f16x8 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_i32x4:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i32x4 i32x4 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_f32x4:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f32x4 f32x4 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_i64x2:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i64x2 i64x2 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_f64x2:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f64x2 f64x2 "v[0:3]" x: i32 "v4", y: ptr "s[0:1]", "global_load_b128");

// CHECK-LABEL: v0_i32x5:
// CHECK: #ASMSTART
// CHECK: image_load v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_1D tfe
// CHECK: #ASMEND
check_reg!(v0_i32x5 i32x5 "v[0:4]" x: i32 "v5", y: i32x8 "s[0:7]", "image_load",
    " dmask:0xf dim:SQ_RSRC_IMG_1D tfe");

// CHECK-LABEL: v0_f32x5:
// CHECK: #ASMSTART
// CHECK: image_load v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_1D tfe
// CHECK: #ASMEND
check_reg!(v0_f32x5 f32x5 "v[0:4]" x: i32 "v5", y: i32x8 "s[0:7]", "image_load",
    " dmask:0xf dim:SQ_RSRC_IMG_1D tfe");

// gfx11-LABEL: v0_i32x6:
// gfx11: #ASMSTART
// gfx11: image_sample_d v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_i32x6 i32x4 "v[0:3]" x: i32x6 "v[4:9]", y: i32x8 "s[0:7]", z: i32x4 "s[8:11]",
    "image_sample_d", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: v0_f32x6:
// gfx11: #ASMSTART
// gfx11: image_sample_d v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_f32x6 i32x4 "v[0:3]" x: f32x6 "v[4:9]", y: i32x8 "s[0:7]", z: i32x4 "s[8:11]",
    "image_sample_d", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: v0_i32x7:
// gfx11: #ASMSTART
// gfx11: image_sample_d_cl v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_i32x7 i32x4 "v[0:3]" x: i32x7 "v[4:10]", y: i32x8 "s[0:7]", z: i32x4 "s[8:11]",
    "image_sample_d_cl", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// gfx11-LABEL: v0_f32x7:
// gfx11: #ASMSTART
// gfx11: image_sample_d_cl v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0xf dim:SQ_RSRC_IMG_2D
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_f32x7 i32x4 "v[0:3]" x: f32x7 "v[4:10]", y: i32x8 "s[0:7]", z: i32x4 "s[8:11]",
    "image_sample_d_cl", " dmask:0xf dim:SQ_RSRC_IMG_2D");

// CHECK-LABEL: s0_i16x16:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i16x16 i16x16 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// CHECK-LABEL: s0_f16x16:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f16x16 f16x16 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// CHECK-LABEL: s0_i32x8:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i32x8 i32x8 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// CHECK-LABEL: s0_f32x8:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f32x8 f32x8 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// CHECK-LABEL: s0_i64x4:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i64x4 i64x4 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// CHECK-LABEL: s0_f64x4:
// CHECK: #ASMSTART
// CHECK: s_load_b256 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f64x4 f64x4 "s[0:7]" x: ptr "s[8:9]", y: i32 "s10", "s_load_b256");

// gfx11-LABEL: v0_i16x16:
// gfx11: #ASMSTART
// gfx11: v_wmma_f32_16x16x16_bf16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_i16x16 f32x8 "v[0:7]" x: i32x8 "v[8:15]", y: i16x16 "v[16:23]", z: f32x8 "v[24:31]",
    "v_wmma_f32_16x16x16_bf16");

// gfx11-LABEL: v0_f16x16:
// gfx11: #ASMSTART
// gfx11: v_wmma_f32_16x16x16_f16 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_f16x16 f32x8 "v[0:7]" x: f16x16 "v[8:15]", y: f16x16 "v[16:23]", z: f32x8 "v[24:31]",
    "v_wmma_f32_16x16x16_f16");

// gfx11-LABEL: v0_i32x8:
// gfx11: #ASMSTART
// gfx11: v_wmma_i32_16x16x16_iu8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_i32x8 i32x8 "v[0:7]" x: i32x4 "v[8:11]", y: i32x4 "v[16:19]", z: i32x8 "v[24:31]",
    "v_wmma_i32_16x16x16_iu8");

// gfx12-LABEL: v0_f32x8:
// gfx12: #ASMSTART
// gfx12: v_wmma_f32_16x16x16_fp8_fp8 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(v0_f32x8 f32x8 "v[0:7]" x: f32x2 "v[8:9]", y: f32x2 "v[16:17]", z: f32x8 "v[24:31]",
    "v_wmma_f32_16x16x16_fp8_fp8");

// gfx12-LABEL: v0_i32x10:
// gfx12: #ASMSTART
// gfx12: image_bvh8_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(v0_i32x10 i32x10 "v[0:9]" "image_bvh8_intersect_ray",
    ", [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]");

// gfx12-LABEL: v0_f32x10:
// gfx12: #ASMSTART
// gfx12: image_bvh8_intersect_ray v{{\[[0-9]+:[0-9]+\]}}, [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(v0_f32x10 f32x10 "v[0:9]" "image_bvh8_intersect_ray",
    ", [v[0:1], v[2:3], v[16:18], v[19:21], v9], s[0:3]");

// CHECK-LABEL: s0_i16x32:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i16x32 i16x32 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");

// CHECK-LABEL: s0_f16x32:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f16x32 f16x32 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");

// CHECK-LABEL: s0_i32x16:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i32x16 i32x16 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");

// CHECK-LABEL: s0_f32x16:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f32x16 f32x16 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");

// CHECK-LABEL: s0_i64x8:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i64x8 i64x8 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");

// CHECK-LABEL: s0_f64x8:
// CHECK: #ASMSTART
// CHECK: s_load_b512 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f64x8 f64x8 "s[0:15]" x: ptr "s[16:17]", y: i32 "s18", "s_load_b512");
