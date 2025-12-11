//@ add-minicore
//@ revisions: gfx11 gfx12
//@ assembly-output: emit-asm
//@ compile-flags: --target amdgcn-amd-amdhsa
//@[gfx11] compile-flags: -Ctarget-cpu=gfx1100
//@[gfx12] compile-flags: -Ctarget-cpu=gfx1200
//@ needs-llvm-components: amdgpu
//@ needs-rust-lld

#![feature(abi_gpu_kernel, no_core, asm_experimental_arch, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

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
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func($($arg_name: $arg_ty,)*) {
            let result: $ret_ty;
            asm!(concat!($mov, " {}", $(", {", stringify!($arg_name), "}",)*),
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
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $func($($arg_name: $arg_ty,)*) {
            let result: $ret_ty;
            asm!(concat!($mov, " ", $ret_reg, $(", ", $arg_reg,)*), lateout($ret_reg) result,
                $(in($arg_reg) $arg_name,)*);
        }
    };
}

// CHECK-LABEL: sgpr_i16:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check!(sgpr_i16 i32 sgpr x: i16 sgpr, y: i16 sgpr, "s_pack_ll_b32_b16");

// gfx11-LABEL: vgpr_i16:
// gfx11: #ASMSTART
// gfx11: v_mov_b16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_i16 i16 vgpr "v_mov_b16");

// gfx12-LABEL: sgpr_f16:
// gfx12: #ASMSTART
// gfx12: s_add_f16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check!(sgpr_f16 f16 sgpr x: f16 sgpr, y: f16 sgpr, "s_add_f16");

// gfx11-LABEL: vgpr_f16:
// gfx11: #ASMSTART
// gfx11: v_mov_b16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check!(vgpr_f16 f16 vgpr "v_mov_b16");

// CHECK-LABEL: sgpr_i32:
// CHECK: #ASMSTART
// CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i32 i32 sgpr "s_mov_b32");

// CHECK-LABEL: vgpr_i32:
// CHECK: #ASMSTART
// CHECK: v_mov_b32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #ASMEND
check!(vgpr_i32 i32 vgpr "v_mov_b32");

// CHECK-LABEL: sgpr_f32:
// CHECK: #ASMSTART
// CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_f32 f32 sgpr "s_mov_b32");

// CHECK-LABEL: vgpr_f32:
// CHECK: #ASMSTART
// CHECK: v_mov_b32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #ASMEND
check!(vgpr_f32 f32 vgpr "v_mov_b32");

// CHECK-LABEL: sgpr_i64:
// CHECK: #ASMSTART
// CHECK: s_mov_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(sgpr_i64 i64 sgpr "s_mov_b64");

// CHECK-LABEL: vgpr_i64:
// CHECK: #ASMSTART
// CHECK: v_lshlrev_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i64 i64 vgpr x: i32 vgpr, y: i64 vgpr, "v_lshlrev_b64");

// CHECK-LABEL: sgpr_f64:
// CHECK: #ASMSTART
// CHECK: s_mov_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(sgpr_f64 f64 sgpr "s_mov_b64");

// CHECK-LABEL: vgpr_f64:
// CHECK: #ASMSTART
// CHECK: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_f64 f64 vgpr x: f64 vgpr, y: f64 vgpr, "v_add_f64");

// CHECK-LABEL: sgpr_i128:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check!(sgpr_i128 i128 sgpr x: ptr sgpr, y: i32 sgpr, "s_load_b128");

// CHECK-LABEL: vgpr_i128:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check!(vgpr_i128 i128 vgpr x: i32 vgpr, y: ptr sgpr, "global_load_b128");

// CHECK-LABEL: s0_i16:
// CHECK: #ASMSTART
// CHECK: s_pack_ll_b32_b16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// CHECK: #ASMEND
check_reg!(s0_i16 i32 "s0" x: i16 "s1", y: i16 "s2", "s_pack_ll_b32_b16");

// gfx11-LABEL: v0_i16:
// gfx11: #ASMSTART
// gfx11: v_mov_b16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_i16 i16 "v0.l" "v_mov_b16");

// gfx12-LABEL: s0_f16:
// gfx12: #ASMSTART
// gfx12: s_add_f16 s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}, s{{[a-z0-9.]+}}
// gfx12: #ASMEND
#[cfg(gfx12)]
check_reg!(s0_f16 f16 "s0" x: f16 "s1", y: f16 "s2", "s_add_f16");

// gfx11-LABEL: v0_f16:
// gfx11: #ASMSTART
// gfx11: v_mov_b16 v{{[a-z0-9.]+}}, v{{[a-z0-9.]+}}
// gfx11: #ASMEND
#[cfg(gfx11)]
check_reg!(v0_f16 f16 "v0.l" "v_mov_b16");

// CHECK-LABEL: s0_i32:
// CHECK: #ASMSTART
// CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i32 i32 "s0" "s_mov_b32");

// CHECK-LABEL: v0_i32:
// CHECK: #ASMSTART
// CHECK: v_mov_b32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(v0_i32 i32 "v0" "v_mov_b32");

// CHECK-LABEL: s0_f32:
// CHECK: #ASMSTART
// CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_f32 f32 "s0" "s_mov_b32");

// CHECK-LABEL: v0_f32:
// CHECK: #ASMSTART
// CHECK: v_mov_b32 v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(v0_f32 f32 "v0" "v_mov_b32");

// CHECK-LABEL: s0_i64:
// CHECK: #ASMSTART
// CHECK: s_mov_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(s0_i64 i64 "s[0:1]" "s_mov_b64");

// CHECK-LABEL: v0_i64:
// CHECK: #ASMSTART
// CHECK: v_lshlrev_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i64 i64 "v[0:1]" x: i32 "v0", y: i64 "v[0:1]", "v_lshlrev_b64");

// CHECK-LABEL: s0_f64:
// CHECK: #ASMSTART
// CHECK: s_mov_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(s0_f64 f64 "s[0:1]" "s_mov_b64");

// CHECK-LABEL: v0_f64:
// CHECK: #ASMSTART
// CHECK: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_f64 f64 "v[0:1]" x: f64 "v[0:1]", y: f64 "v[2:3]", "v_add_f64");

// CHECK-LABEL: s0_i128:
// CHECK: #ASMSTART
// CHECK: s_load_b128 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
// CHECK: #ASMEND
check_reg!(s0_i128 i128 "s[0:3]" x: ptr "s[0:1]", y: i32 "s0", "s_load_b128");

// CHECK-LABEL: v0_i128:
// CHECK: #ASMSTART
// CHECK: global_load_b128 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
// CHECK: #ASMEND
check_reg!(v0_i128 i128 "v[0:3]" x: i32 "v0", y: ptr "s[0:1]", "global_load_b128");
