#![no_std]
#![feature(
    const_refs_to_cell,
    const_maybe_uninit_as_mut_ptr,
    const_mut_refs,
    convert_float_to_int,
    decl_macro,
    intra_doc_pointers,
    platform_intrinsics,
    repr_simd,
    simd_ffi,
    staged_api,
    stdsimd,
    strict_provenance,
    ptr_metadata
)]
#![cfg_attr(feature = "generic_const_exprs", feature(generic_const_exprs))]
#![cfg_attr(feature = "generic_const_exprs", allow(incomplete_features))]
#![warn(missing_docs, clippy::missing_inline_in_public_items)] // basically all items, really
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![unstable(feature = "portable_simd", issue = "86656")]
//! Portable SIMD module.

#[path = "mod.rs"]
mod core_simd;
pub use self::core_simd::simd;
