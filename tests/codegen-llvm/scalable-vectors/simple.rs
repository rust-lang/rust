//@ edition: 2021
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(simd_ffi, rustc_attrs, link_llvm_intrinsics, stdarch_aarch64_sve)]

// ============================================================================
// 演示 SVE 类型的多种导入方式
// ============================================================================

// 方式 1: 直接从 aarch64 模块导入所有类型（推荐，因为类型已被重新导出）
use std::arch::aarch64::{svint32_t, svint64_t, svfloat32_t, svuint32_t};

// 方式 1b: 通过 sve 模块导入所有类型（需要 sve 模块被公开导出）
// use std::arch::aarch64::sve::types::*;

// 方式 2: 导入 types 模块，然后使用模块路径
// use std::arch::aarch64::sve::types;
// 使用: types::svint32_t, types::svint64_t 等

// 方式 3: 从 types 模块导入特定类型
// use std::arch::aarch64::sve::types::{svint32_t, svint64_t, svfloat32_t};

// 方式 4: 直接导入类型（原有方式，仍然有效）
// use std::arch::aarch64::{svint32_t, svint64_t, svfloat32_t};

// 方式 5: 通过 aarch64 模块导入（因为 sve 模块被重新导出）
// use std::arch::aarch64::types::*;

#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s32(op: i32) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
    }
    unsafe { _svdup_n_s32(op) }
}

#[inline]
#[target_feature(enable = "sve,sve2")]
pub unsafe fn svxar_n_s32<const IMM3: i32>(op1: svint32_t, op2: svint32_t) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.xar.nxv4i32")]
        fn _svxar_n_s32(op1: svint32_t, op2: svint32_t, imm3: i32) -> svint32_t;
    }
    unsafe { _svxar_n_s32(op1, op2, IMM3) }
}

#[inline(never)]
#[no_mangle]
#[target_feature(enable = "sve,sve2")]
// CHECK: define <vscale x 4 x i32> @pass_as_ref(ptr noalias noundef readonly align 16 captures(none) dereferenceable(16) %a, <vscale x 4 x i32> %b)
pub unsafe fn pass_as_ref(a: &svint32_t, b: svint32_t) -> svint32_t {
    // CHECK: load <vscale x 4 x i32>, ptr %a, align 16
    svxar_n_s32::<1>(*a, b)
}

#[no_mangle]
#[target_feature(enable = "sve,sve2")]
// CHECK: define <vscale x 4 x i32> @test()
pub unsafe fn test() -> svint32_t {
    let a = svdup_n_s32(1);
    let b = svdup_n_s32(2);
    // CHECK: %_0 = call <vscale x 4 x i32> @pass_as_ref(ptr noalias noundef nonnull readonly align 16 dereferenceable(16) %a, <vscale x 4 x i32> %b)
    pass_as_ref(&a, b)
}

// ============================================================================
// 演示使用不同类型的示例（展示 types 模块导入的便利性）
// ============================================================================

// 示例：使用 svint64_t 类型
#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_s64(op: i64) -> svint64_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv2i64")]
        fn _svdup_n_s64(op: i64) -> svint64_t;
    }
    unsafe { _svdup_n_s64(op) }
}

// 示例：使用 svfloat32_t 类型
#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_f32(op: f32) -> svfloat32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4f32")]
        fn _svdup_n_f32(op: f32) -> svfloat32_t;
    }
    unsafe { _svdup_n_f32(op) }
}

// 示例：混合使用多种类型（注意：可扩展向量类型不能作为元组字段）
// 因此我们分别创建三个函数来演示不同类型的使用
#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_i32() -> svint32_t {
    let i32_vec = svdup_n_s32(42);
    i32_vec
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_i64() -> svint64_t {
    let i64_vec = svdup_n_s64(100);
    i64_vec
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_f32() -> svfloat32_t {
    let f32_vec = svdup_n_f32(3.14);
    f32_vec
}

// 示例：使用 svuint32_t 类型（展示无符号类型）
#[inline(never)]
#[target_feature(enable = "sve")]
pub unsafe fn svdup_n_u32(op: u32) -> svuint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_u32(op: u32) -> svuint32_t;
    }
    unsafe { _svdup_n_u32(op) }
}
