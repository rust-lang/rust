#![doc = include_str!("core_arch_docs.md")]
#![allow(improper_ctypes_definitions)]
#![allow(dead_code)]
#![allow(unused_features)]
#![allow(internal_features)]
#![allow(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]
#![feature(
    custom_inner_attributes,
    link_llvm_intrinsics,
    repr_simd,
    simd_ffi,
    proc_macro_hygiene,
    stmt_expr_attributes,
    core_intrinsics,
    no_core,
    fmt_helpers_for_derive,
    rustc_attrs,
    staged_api,
    doc_cfg,
    riscv_target_feature,
    arm_target_feature,
    mips_target_feature,
    powerpc_target_feature,
    s390x_target_feature,
    loongarch_target_feature,
    wasm_target_feature,
    abi_unadjusted,
    rtm_target_feature,
    allow_internal_unstable,
    decl_macro,
    asm_experimental_arch,
    x86_amx_intrinsics,
    f16,
    aarch64_unstable_target_feature,
    bigint_helper_methods,
    funnel_shifts
)]
#![cfg_attr(test, feature(test, abi_vectorcall, stdarch_internal))]
#![deny(clippy::missing_inline_in_public_items)]
#![allow(
    clippy::identity_op,
    clippy::inline_always,
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cognitive_complexity,
    clippy::many_single_char_names,
    clippy::missing_safety_doc,
    clippy::shadow_reuse,
    clippy::similar_names,
    clippy::unusual_byte_groupings,
    clippy::wrong_self_convention
)]
#![cfg_attr(test, allow(unused_imports))]
#![no_std]
#![stable(feature = "stdsimd", since = "1.27.0")]
#![doc(
    test(attr(deny(warnings))),
    test(attr(allow(dead_code, deprecated, unused_variables, unused_mut)))
)]
#![cfg_attr(
    test,
    feature(
        stdarch_arm_feature_detection,
        stdarch_powerpc_feature_detection,
        stdarch_s390x_feature_detection
    )
)]

#[cfg(test)]
#[macro_use]
extern crate std;

#[path = "mod.rs"]
mod core_arch;

#[stable(feature = "stdsimd", since = "1.27.0")]
pub mod arch {
    #[stable(feature = "stdsimd", since = "1.27.0")]
    #[allow(unused_imports)]
    pub use crate::core_arch::arch::*;
    #[stable(feature = "stdsimd", since = "1.27.0")]
    pub use core::arch::asm;
}

#[allow(unused_imports)]
use core::{array, convert, ffi, fmt, hint, intrinsics, marker, mem, ops, ptr, sync};
