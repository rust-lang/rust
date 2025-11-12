//@ edition: 2021
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(simd_ffi, rustc_attrs, link_llvm_intrinsics, stdarch_aarch64_sve)]

// ============================================================================
// 演示 SVE 类型和函数的导入方式
// ============================================================================

// 从 aarch64 模块导入 SVE 类型（类型已被重新导出）
use std::arch::aarch64::{svint32_t, svint64_t, svfloat32_t, svuint32_t, svpattern};

// 从 aarch64 模块导入 SVE intrinsics 函数（函数已被重新导出）
use std::arch::aarch64::{
    svdup_n_s32, svdup_n_s64, svdup_n_f32, svdup_n_u32,
    svadd_s32_z, svsub_s32_z, svmul_s32_z,
    svptrue_pat_b32,
};

// 注意：svxar_n_s32 是 SVE2 函数，如果库中未定义，可以保留本地定义
// 或者使用其他已定义的函数替代
#[inline]
#[target_feature(enable = "sve,sve2")]
pub unsafe fn svxar_n_s32<const IMM3: i32>(op1: svint32_t, op2: svint32_t) -> svint32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.xar.nxv4i32")]
        fn _svxar_n_s32(op1: svint32_t, op2: svint32_t, imm3: i32) -> svint32_t;
    }
    unsafe { _svxar_n_s32(op1, op2, IMM3) }
}

// ============================================================================
// 测试用例：使用库中定义的 SVE intrinsics 函数
// ============================================================================

#[inline(never)]
#[no_mangle]
#[target_feature(enable = "sve,sve2")]
// CHECK: define <vscale x 4 x i32> @pass_as_ref(ptr noalias noundef readonly align 16 captures(none) dereferenceable(16) %a, <vscale x 4 x i32> %b)
pub unsafe fn pass_as_ref(a: &svint32_t, b: svint32_t) -> svint32_t {
    // CHECK: load <vscale x 4 x i32>, ptr %a, align 16
    svxar_n_s32::<1>(*a, b)
}

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK: define <vscale x 4 x i32> @test()
pub unsafe fn test() -> svint32_t {
    // 使用库中定义的 svdup_n_s32 函数
    let a = svdup_n_s32(1);
    let b = svdup_n_s32(2);
    // CHECK: %_0 = call <vscale x 4 x i32> @pass_as_ref(ptr noalias noundef nonnull readonly align 16 dereferenceable(16) %a, <vscale x 4 x i32> %b)
    pass_as_ref(&a, b)
}

// ============================================================================
// 演示使用不同类型的示例
// ============================================================================

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_i32() -> svint32_t {
    // 使用库中定义的 svdup_n_s32 函数
    let i32_vec = svdup_n_s32(42);
    i32_vec
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_i64() -> svint64_t {
    // 使用库中定义的 svdup_n_s64 函数
    let i64_vec = svdup_n_s64(100);
    i64_vec
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_f32() -> svfloat32_t {
    // 使用库中定义的 svdup_n_f32 函数
    let f32_vec = svdup_n_f32(3.14);
    f32_vec
}

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_multiple_types_u32() -> svuint32_t {
    // 使用库中定义的 svdup_n_u32 函数
    let u32_vec = svdup_n_u32(200);
    u32_vec
}

// ============================================================================
// 演示使用其他 SVE intrinsics 函数
// ============================================================================

#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe fn test_arithmetic_operations() -> svint32_t {
    // 使用库中定义的函数进行算术运算
    let a = svdup_n_s32(10);
    let b = svdup_n_s32(20);
    // 创建全真谓词（使用 SV_ALL 模式）
    const PATTERN_ALL: svpattern = svpattern::SV_ALL;
    let pg = svptrue_pat_b32::<PATTERN_ALL>();
    // 加法
    let sum = svadd_s32_z(pg, a, b);
    // 减法
    let diff = svsub_s32_z(pg, b, a);
    // 乘法
    svmul_s32_z(pg, sum, diff)
}
