//! SIMD and vendor intrinsics support library.
//!
//! This documentation is only for one particular architecture, you can find
//! others at:
//!
//! * [i686](https://rust-lang-nursery.github.io/stdsimd/i686/stdsimd/)
//! * [`x86_64`](https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/)
//! * [arm](https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/)
//! * [aarch64](https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/)

#![cfg_attr(feature = "strict", deny(warnings))]
#![allow(dead_code)]
#![allow(unused_features)]
#![feature(const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd,
           simd_ffi, target_feature, cfg_target_feature, i128_type, asm,
           integer_atomics, stmt_expr_attributes, core_intrinsics,
           crate_in_paths, no_core, attr_literals, rustc_attrs)]
#![cfg_attr(test, feature(proc_macro, test, attr_literals, abi_vectorcall))]
#![cfg_attr(feature = "cargo-clippy",
            allow(inline_always, too_many_arguments, cast_sign_loss,
                  cast_lossless, cast_possible_wrap,
                  cast_possible_truncation, cast_precision_loss,
                  shadow_reuse, cyclomatic_complexity, similar_names,
                  many_single_char_names))]
#![cfg_attr(test, allow(unused_imports))]
#![no_core]

#[cfg_attr(not(test), macro_use)]
extern crate core as _core;
#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate stdsimd_test;
#[cfg(test)]
extern crate test;
#[cfg(test)]
#[macro_use]
extern crate stdsimd;

#[path = "../../../coresimd/mod.rs"]
mod coresimd;

pub use coresimd::simd;

pub mod arch {
    #[cfg(target_arch = "x86")]
    pub mod x86 { pub use coresimd::vendor::*; }
    #[cfg(target_arch = "x86_64")]
    pub mod x86_64 { pub use coresimd::vendor::*; }
    #[cfg(target_arch = "arm")]
    pub mod arm { pub use coresimd::vendor::*; }
    #[cfg(target_arch = "aarch64")]
    pub mod aarch64 { pub use coresimd::vendor::*; }
}

#[allow(unused_imports)]
use _core::clone;
#[allow(unused_imports)]
use _core::cmp;
#[allow(unused_imports)]
use _core::convert;
#[allow(unused_imports)]
use _core::fmt;
#[allow(unused_imports)]
use _core::intrinsics;
#[allow(unused_imports)]
use _core::iter;
#[allow(unused_imports)]
use _core::marker;
#[allow(unused_imports)]
use _core::mem;
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
