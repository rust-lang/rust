//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(auto_traits)]
#![feature(never_type)]
#![feature(associated_type_bounds)]
#![feature(min_specialization)]
#![feature(core_intrinsics)]
#![feature(maybe_uninit_slice)]
#![feature(let_else)]
#![feature(negative_impls)]
#![feature(new_uninit)]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod collection_impls;
mod serialize;

pub mod leb128;
pub mod opaque;

/// This trait is used to mark datastructures as safe for Mmap.
pub unsafe auto trait MmapSafe {}
impl<'a, T> !MmapSafe for &'a T {}
impl<'a, T> !MmapSafe for &'a mut T {}
impl<T> !MmapSafe for *const T {}
impl<T> !MmapSafe for *mut T {}
