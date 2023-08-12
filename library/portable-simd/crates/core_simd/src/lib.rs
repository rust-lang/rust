#![no_std]
// tidy-alphabetical-start
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(const_mut_refs)]
#![feature(const_refs_to_cell)]
#![feature(convert_float_to_int)]
#![feature(decl_macro)]
#![feature(intra_doc_pointers)]
#![feature(platform_intrinsics)]
#![feature(ptr_metadat)]
#![feature(repr_simd)]
#![feature(simd_ffi)]
#![feature(staged_api)]
#![feature(stdsimd)]
#![feature(strict_provenance)]
// tidy-alphabetical-end
#![cfg_attr(feature = "generic_const_exprs", feature(generic_const_exprs))]
#![cfg_attr(feature = "generic_const_exprs", allow(incomplete_features))]
#![warn(missing_docs, clippy::missing_inline_in_public_items)] // basically all items, really
#![deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)]
#![unstable(feature = "portable_simd", issue = "86656")]
//! Portable SIMD module.

#[path = "mod.rs"]
mod core_simd;
pub use self::core_simd::simd;
