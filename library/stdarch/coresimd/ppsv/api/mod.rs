//! This module defines the API of portable vector types.
//!
//! # API
//!
//! ## Traits
//!
//! All portable vector types implement the following traits:
//!
//! * [x] `Copy`,
//! * [x] `Clone`,
//! * [x] `Debug`,
//! * [x] `Default`
//! * [x] `PartialEq`
//! * [x] `PartialOrd` (TODO: re-write in term of
//!        comparison operations and boolean reductions),
//!
//! Non-floating-point vector types also implement:
//!
//! * [x] `Hash`,
//! * [x] `Eq`, and
//! * [x] `Ord`.
//!
//! Integer vector types also implement:
//!
//! * [x] `fmt::LowerHex`.
//!
//! ## Conversions
//!
//! * [x]: `FromBits/IntoBits`: bitwise lossless transmutes between vectors of
//!        the same size (i.e., same `mem::size_of`).
//! * [x]: `From/Into`: casts between vectors with the same number of lanes
//!        (potentially lossy).
//!
//! ## Inherent methods
//!
//! * [x] minimal API: implemented by all vector types except for boolean
//!       vectors.
//! * [x] minimal boolean vector API: implemented by boolean vectors.
//! * [x] load/store API: aligned and unaligned memory loads and
//!       stores - implemented by all vectors.
//! * [x] comparison API: vector lane-wise comparison producing
//!       boolean vectors - implemented by all vectors.
//! * [x] arithmetic operations: implemented by all non-boolean vectors.
//! * [x] `std::ops::Neg`: implemented by signed-integer and floating-point
//!       vectors.
//! * [x] bitwise operations: implemented by integer and boolean
//!       vectors.
//! * [x] shift operations: implemented by integer vectors.
//! * [x] arithmetic reductions: implemented by integer and floating-point
//!       vectors.
//! * [x] bitwise reductions: implemented by integer and boolean
//!       vectors.
//! * [x] boolean reductions: implemented by boolean vectors.
//! * [ ] portable shuffles: `shufflevector`.
//! * [ ] portable `gather`/`scatter`:

/// Adds the vector type `$id`, with elements of types `$elem_tys`.
macro_rules! define_ty {
    ($id:ident, $($elem_tys:ident),+ | $(#[$doc:meta])*) => {
        $(#[$doc])*
            #[repr(simd)]
        #[derive(Copy, Debug, /*FIXME:*/ PartialOrd)]
        #[allow(non_camel_case_types)]
        pub struct $id($($elem_tys),*);
    }
}

#[macro_use]
mod arithmetic_ops;
#[macro_use]
mod arithmetic_reductions;
#[macro_use]
mod bitwise_ops;
#[macro_use]
mod bitwise_reductions;
#[macro_use]
mod boolean_reductions;
#[macro_use]
mod bool_vectors;
#[macro_use]
mod cmp;
#[macro_use]
mod default;
#[macro_use]
mod eq;
#[macro_use]
mod fmt;
#[macro_use]
mod from;
#[macro_use]
mod from_bits;
#[macro_use]
mod hash;
#[macro_use]
mod load_store;
#[macro_use]
mod minimal;
#[macro_use]
mod minmax_reductions;
#[macro_use]
mod neg;
#[macro_use]
mod partial_eq;
// TODO:
//#[macro_use]
//mod partial_ord;
// TODO:
//#[macro_use]
//mod shuffles;
// TODO:
//#[macro_use]
//mod gather_scatter;
#[macro_use]
mod shifts;

/// Imports required to implement vector types using the macros.

macro_rules! simd_api_imports {
    () => {
        use ::coresimd::simd_llvm::*;
        use fmt;
        use hash;
        use ops;
        #[allow(unused_imports)]
        use num;
        use cmp::{Eq, PartialEq};
        use ptr;
        use mem;
        #[allow(unused_imports)]
        use convert::{From, Into};
        use slice::SliceExt;
        #[allow(unused_imports)]
        use iter::Iterator;
        #[allow(unused_imports)]
        use default::Default;
        use clone::Clone;
        use super::codegen::sum::{ReduceAdd};
        use super::codegen::product::{ReduceMul};
        use super::codegen::and::{ReduceAnd};
        use super::codegen::or::{ReduceOr};
        use super::codegen::xor::{ReduceXor};
        use super::codegen::min::{ReduceMin};
        use super::codegen::max::{ReduceMax};
    }
}

/// Defines a portable packed SIMD floating-point vector type.
macro_rules! simd_f_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $bool_ty:ident, $test_mod:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        define_ty!($id, $($elem_tys),+ | $(#[$doc])*);
        impl_minimal!($id, $elem_ty, $elem_count, $($elem_name),*);
        impl_load_store!($id, $elem_ty, $elem_count);
        impl_cmp!($id, $bool_ty);
        impl_arithmetic_ops!($id);
        impl_arithmetic_reductions!($id, $elem_ty);
        impl_minmax_reductions!($id, $elem_ty);
        impl_neg_op!($id, $elem_ty);
        impl_partial_eq!($id);
        impl_default!($id, $elem_ty);

        #[cfg(test)]
        mod $test_mod {
            test_minimal!($id, $elem_ty, $elem_count);
            test_load_store!($id, $elem_ty);
            test_cmp!($id, $elem_ty, $bool_ty, 1. as $elem_ty, 0. as $elem_ty);
            test_arithmetic_ops!($id, $elem_ty);
            test_arithmetic_reductions!($id, $elem_ty);
            test_minmax_reductions!($id, $elem_ty);
            test_neg_op!($id, $elem_ty);
            test_partial_eq!($id, 1. as $elem_ty, 0. as $elem_ty);
            test_default!($id, $elem_ty);
        }
    }
}

/// Defines a portable packed SIMD signed-integer vector type.
macro_rules! simd_i_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $bool_ty:ident, $test_mod:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        define_ty!($id, $($elem_tys),+ | $(#[$doc])*);
        impl_minimal!($id, $elem_ty, $elem_count, $($elem_name),*);
        impl_load_store!($id, $elem_ty, $elem_count);
        impl_cmp!($id, $bool_ty);
        impl_hash!($id, $elem_ty);
        impl_arithmetic_ops!($id);
        impl_arithmetic_reductions!($id, $elem_ty);
        impl_minmax_reductions!($id, $elem_ty);
        impl_neg_op!($id, $elem_ty);
        impl_bitwise_ops!($id, !(0 as $elem_ty));
        impl_bitwise_reductions!($id, $elem_ty);
        impl_all_shifts!($id, $elem_ty);
        impl_hex_fmt!($id, $elem_ty);
        impl_eq!($id);
        impl_partial_eq!($id);
        impl_default!($id, $elem_ty);

        #[cfg(test)]
        mod $test_mod {
            test_minimal!($id, $elem_ty, $elem_count);
            test_load_store!($id, $elem_ty);
            test_cmp!($id, $elem_ty, $bool_ty, 1 as $elem_ty, 0 as $elem_ty);
            test_hash!($id, $elem_ty);
            test_arithmetic_ops!($id, $elem_ty);
            test_arithmetic_reductions!($id, $elem_ty);
            test_minmax_reductions!($id, $elem_ty);
            test_neg_op!($id, $elem_ty);
            test_int_bitwise_ops!($id, $elem_ty);
            test_bitwise_reductions!($id, !(0 as $elem_ty));
            test_all_shift_ops!($id, $elem_ty);
            test_hex_fmt!($id, $elem_ty);
            test_partial_eq!($id, 1 as $elem_ty, 0 as $elem_ty);
            test_default!($id, $elem_ty);
        }
    }
}

/// Defines a portable packed SIMD unsigned-integer vector type.
macro_rules! simd_u_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $bool_ty:ident, $test_mod:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        define_ty!($id, $($elem_tys),+ | $(#[$doc])*);
        impl_minimal!($id, $elem_ty, $elem_count, $($elem_name),*);
        impl_load_store!($id, $elem_ty, $elem_count);
        impl_cmp!($id, $bool_ty);
        impl_hash!($id, $elem_ty);
        impl_arithmetic_ops!($id);
        impl_arithmetic_reductions!($id, $elem_ty);
        impl_minmax_reductions!($id, $elem_ty);
        impl_bitwise_ops!($id, !(0 as $elem_ty));
        impl_bitwise_reductions!($id, $elem_ty);
        impl_all_shifts!($id, $elem_ty);
        impl_hex_fmt!($id, $elem_ty);
        impl_eq!($id);
        impl_partial_eq!($id);
        impl_default!($id, $elem_ty);

        #[cfg(test)]
        mod $test_mod {
            test_minimal!($id, $elem_ty, $elem_count);
            test_load_store!($id, $elem_ty);
            test_cmp!($id, $elem_ty, $bool_ty, 1 as $elem_ty, 0 as $elem_ty);
            test_hash!($id, $elem_ty);
            test_arithmetic_ops!($id, $elem_ty);
            test_arithmetic_reductions!($id, $elem_ty);
            test_minmax_reductions!($id, $elem_ty);
            test_int_bitwise_ops!($id, $elem_ty);
            test_bitwise_reductions!($id, !(0 as $elem_ty));
            test_all_shift_ops!($id, $elem_ty);
            test_hex_fmt!($id, $elem_ty);
            test_partial_eq!($id, 1 as $elem_ty, 0 as $elem_ty);
            test_default!($id, $elem_ty);
        }
    }
}

/// Defines a portable packed SIMD boolean vector type.
macro_rules! simd_b_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $test_mod:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        define_ty!($id, $($elem_tys),+ | $(#[$doc])*);
        impl_bool_minimal!($id, $elem_ty, $elem_count, $($elem_name),*);
        impl_bitwise_ops!($id, true);
        impl_bool_bitwise_reductions!($id, bool);
        impl_bool_reductions!($id);
        impl_bool_cmp!($id, $id);
        impl_eq!($id);
        impl_partial_eq!($id);
        impl_default!($id, bool);

        #[cfg(test)]
        mod $test_mod {
            test_bool_minimal!($id, $elem_count);
            test_bool_bitwise_ops!($id);
            test_bool_reductions!($id);
            test_bitwise_reductions!($id, true);
            test_cmp!($id, $elem_ty, $id, true, false);
            test_partial_eq!($id, true, false);
            test_default!($id, bool);
        }
    }
}
