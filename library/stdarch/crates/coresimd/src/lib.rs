//! SIMD and vendor intrinsics support library.
//!
//! This documentation is for the `coresimd` crate, but you probably want to
//! use the [`stdsimd` crate][stdsimd] which should have more complete
//! documentation.
//!
//! [stdsimd]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/

#![cfg_attr(feature = "strict", deny(warnings))]
#![allow(dead_code)]
#![allow(unused_features)]
#![feature(const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd,
           simd_ffi, target_feature, cfg_target_feature, asm,
           integer_atomics, stmt_expr_attributes, core_intrinsics,
           crate_in_paths, no_core, attr_literals, rustc_attrs, stdsimd,
           staged_api, fn_must_use, core_float, core_slice_ext, align_offset,
           doc_cfg)]
#![cfg_attr(test,
            feature(proc_macro, test, attr_literals, abi_vectorcall,
                    untagged_unions))]
#![cfg_attr(feature = "cargo-clippy",
            allow(inline_always, too_many_arguments, cast_sign_loss,
                  cast_lossless, cast_possible_wrap,
                  cast_possible_truncation, cast_precision_loss,
                  shadow_reuse, cyclomatic_complexity, similar_names,
                  many_single_char_names))]
#![cfg_attr(test, allow(unused_imports))]
#![no_core]
#![unstable(feature = "stdsimd", issue = "0")]
#![doc(test(attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables,
                       unused_mut))))]

#[cfg_attr(not(test), macro_use)]
extern crate core as _core;
#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
#[macro_use]
extern crate stdsimd;
#[cfg(test)]
extern crate stdsimd_test;
#[cfg(test)]
extern crate test;

macro_rules! test_v16 {
    ($item: item) => {};
}
macro_rules! test_v32 {
    ($item: item) => {};
}
macro_rules! test_v64 {
    ($item: item) => {};
}
macro_rules! test_v128 {
    ($item: item) => {};
}
macro_rules! test_v256 {
    ($item: item) => {};
}
macro_rules! test_v512 {
    ($item: item) => {};
}
macro_rules! vector_impl {
    ($([$f:ident, $($args:tt)*]),*) => { $($f!($($args)*);)* }
}

#[path = "../../../coresimd/mod.rs"]
mod coresimd;

pub use coresimd::arch;
pub use coresimd::simd;

#[allow(unused_imports)]
use _core::clone;
#[allow(unused_imports)]
use _core::cmp;
#[allow(unused_imports)]
use _core::convert;
#[allow(unused_imports)]
use _core::default;
#[allow(unused_imports)]
use _core::fmt;
#[allow(unused_imports)]
use _core::hash;
#[allow(unused_imports)]
use _core::intrinsics;
#[allow(unused_imports)]
use _core::iter;
#[allow(unused_imports)]
use _core::marker;
#[allow(unused_imports)]
use _core::mem;
#[allow(unused_imports)]
use _core::num;
#[allow(unused_imports)]
use _core::ops;
#[allow(unused_imports)]
use _core::option;
#[allow(unused_imports)]
use _core::prelude;
#[allow(unused_imports)]
use _core::ptr;
#[allow(unused_imports)]
use _core::result;
#[allow(unused_imports)]
use _core::slice;
