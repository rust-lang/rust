#![no_std]
#![feature(
    const_intrinsic_copy,
    const_refs_to_cell,
    const_maybe_uninit_as_mut_ptr,
    const_mut_refs,
    convert_float_to_int,
    core_intrinsics,
    decl_macro,
    inline_const,
    intra_doc_pointers,
    repr_simd,
    simd_ffi,
    staged_api,
    strict_provenance,
    ptr_metadata
)]
#![cfg_attr(
    all(
        any(target_arch = "aarch64", target_arch = "arm",),
        any(
            all(target_feature = "v6", not(target_feature = "mclass")),
            all(target_feature = "mclass", target_feature = "dsp"),
        )
    ),
    feature(stdarch_arm_dsp)
)]
#![cfg_attr(
    all(target_arch = "arm", target_feature = "v7"),
    feature(stdarch_arm_neon_intrinsics)
)]
#![cfg_attr(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    feature(stdarch_powerpc)
)]
#![warn(missing_docs, clippy::missing_inline_in_public_items)] // basically all items, really
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![allow(internal_features)]
#![unstable(feature = "portable_simd", issue = "86656")]
//! Portable SIMD module.

#[path = "mod.rs"]
mod core_simd;
pub use self::core_simd::simd;
