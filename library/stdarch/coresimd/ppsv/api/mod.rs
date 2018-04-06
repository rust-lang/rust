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
//! * [x] `PartialOrd` (TODO: tests)
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
#![allow(unused)]

/// Adds the vector type `$id`, with elements of types `$elem_tys`.
macro_rules! define_ty {
    ($id:ident, $($elem_tys:ident),+ | $(#[$doc:meta])*) => {
        $(#[$doc])*
        #[repr(simd)]
        #[derive(Copy, Clone, Debug, /*FIXME:*/ PartialOrd)]
        #[allow(non_camel_case_types)]
        pub struct $id($($elem_tys),*);
    }
}

#[macro_use]
mod arithmetic_ops;
#[macro_use]
mod arithmetic_scalar_ops;
#[macro_use]
mod arithmetic_reductions;
#[macro_use]
mod bitwise_ops;
#[macro_use]
mod bitwise_scalar_ops;
#[macro_use]
mod bitwise_reductions;
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
mod masks;
#[macro_use]
mod masks_reductions;
#[macro_use]
mod minimal;
#[macro_use]
mod minmax;
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
mod masks_select;
#[macro_use]
mod scalar_shifts;
#[macro_use]
mod shifts;

/// Sealed trait used for constraining select implementations.
pub trait Lanes<A> {}

/// Defines a portable packed SIMD floating-point vector type.
macro_rules! simd_f_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $mask_ty:ident, $test_mod:ident, $test_macro:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        vector_impl!(
            [define_ty, $id, $($elem_tys),+ | $(#[$doc])*],
            [impl_minimal, $id, $elem_ty, $elem_count, $($elem_name),*],
            [impl_load_store, $id, $elem_ty, $elem_count],
            [impl_cmp, $id, $mask_ty],
            [impl_arithmetic_ops, $id],
            [impl_arithmetic_scalar_ops, $id, $elem_ty],
            [impl_arithmetic_reductions, $id, $elem_ty],
            [impl_minmax_reductions, $id, $elem_ty],
            [impl_neg_op, $id, $elem_ty],
            [impl_partial_eq, $id],
            [impl_default, $id, $elem_ty],
            [impl_float_minmax_ops, $id]
        );

        $test_macro!(
            #[cfg(test)]
            mod $test_mod {
                test_minimal!($id, $elem_ty, $elem_count);
                test_load_store!($id, $elem_ty);
                test_cmp!($id, $elem_ty, $mask_ty, 1. as $elem_ty, 0. as $elem_ty);
                test_arithmetic_ops!($id, $elem_ty);
                test_arithmetic_scalar_ops!($id, $elem_ty);
                test_arithmetic_reductions!($id, $elem_ty);
                test_minmax_reductions!($id, $elem_ty);
                test_neg_op!($id, $elem_ty);
                test_partial_eq!($id, 1. as $elem_ty, 0. as $elem_ty);
                test_default!($id, $elem_ty);
                test_mask_select!($mask_ty, $id, $elem_ty);
                test_float_minmax_ops!($id, $elem_ty);
            }
        );
    }
}

/// Defines a portable packed SIMD signed-integer vector type.
macro_rules! simd_i_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $mask_ty:ident, $test_mod:ident, $test_macro:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        vector_impl!(
            [define_ty, $id, $($elem_tys),+ | $(#[$doc])*],
            [impl_minimal, $id, $elem_ty, $elem_count, $($elem_name),*],
            [impl_load_store, $id, $elem_ty, $elem_count],
            [impl_cmp, $id, $mask_ty],
            [impl_hash, $id, $elem_ty],
            [impl_arithmetic_ops, $id],
            [impl_arithmetic_scalar_ops, $id, $elem_ty],
            [impl_arithmetic_reductions, $id, $elem_ty],
            [impl_minmax_reductions, $id, $elem_ty],
            [impl_neg_op, $id, $elem_ty],
            [impl_bitwise_ops, $id, !(0 as $elem_ty)],
            [impl_bitwise_scalar_ops, $id, $elem_ty],
            [impl_bitwise_reductions, $id, $elem_ty],
            [impl_all_scalar_shifts, $id, $elem_ty],
            [impl_vector_shifts, $id, $elem_ty],
            [impl_hex_fmt, $id, $elem_ty],
            [impl_eq, $id],
            [impl_partial_eq, $id],
            [impl_default, $id, $elem_ty],
            [impl_int_minmax_ops, $id]
        );

        $test_macro!(
            #[cfg(test)]
            mod $test_mod {
                test_minimal!($id, $elem_ty, $elem_count);
                test_load_store!($id, $elem_ty);
                test_cmp!($id, $elem_ty, $mask_ty, 1 as $elem_ty, 0 as $elem_ty);
                test_hash!($id, $elem_ty);
                test_arithmetic_ops!($id, $elem_ty);
                test_arithmetic_scalar_ops!($id, $elem_ty);
                test_arithmetic_reductions!($id, $elem_ty);
                test_minmax_reductions!($id, $elem_ty);
                test_neg_op!($id, $elem_ty);
                test_int_bitwise_ops!($id, $elem_ty);
                test_int_bitwise_scalar_ops!($id, $elem_ty);
                test_bitwise_reductions!($id, !(0 as $elem_ty));
                test_all_scalar_shift_ops!($id, $elem_ty);
                test_vector_shift_ops!($id, $elem_ty);
                test_hex_fmt!($id, $elem_ty);
                test_partial_eq!($id, 1 as $elem_ty, 0 as $elem_ty);
                test_default!($id, $elem_ty);
                test_mask_select!($mask_ty, $id, $elem_ty);
                test_int_minmax_ops!($id, $elem_ty);
            }
        );
    }
}

/// Defines a portable packed SIMD unsigned-integer vector type.
macro_rules! simd_u_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $mask_ty:ident, $test_mod:ident, $test_macro:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        vector_impl!(
            [define_ty, $id, $($elem_tys),+ | $(#[$doc])*],
            [impl_minimal, $id, $elem_ty, $elem_count, $($elem_name),*],
            [impl_load_store, $id, $elem_ty, $elem_count],
            [impl_cmp, $id, $mask_ty],
            [impl_hash, $id, $elem_ty],
            [impl_arithmetic_ops, $id],
            [impl_arithmetic_scalar_ops, $id, $elem_ty],
            [impl_arithmetic_reductions, $id, $elem_ty],
            [impl_minmax_reductions, $id, $elem_ty],
            [impl_bitwise_scalar_ops, $id, $elem_ty],
            [impl_bitwise_ops, $id, !(0 as $elem_ty)],
            [impl_bitwise_reductions, $id, $elem_ty],
            [impl_all_scalar_shifts, $id, $elem_ty],
            [impl_vector_shifts, $id, $elem_ty],
            [impl_hex_fmt, $id, $elem_ty],
            [impl_eq, $id],
            [impl_partial_eq, $id],
            [impl_default, $id, $elem_ty],
            [impl_int_minmax_ops, $id]
        );

        $test_macro!(
            #[cfg(test)]
            mod $test_mod {
                test_minimal!($id, $elem_ty, $elem_count);
                test_load_store!($id, $elem_ty);
                test_cmp!($id, $elem_ty, $mask_ty, 1 as $elem_ty, 0 as $elem_ty);
                test_hash!($id, $elem_ty);
                test_arithmetic_ops!($id, $elem_ty);
                test_arithmetic_scalar_ops!($id, $elem_ty);
                test_arithmetic_reductions!($id, $elem_ty);
                test_minmax_reductions!($id, $elem_ty);
                test_int_bitwise_ops!($id, $elem_ty);
                test_int_bitwise_scalar_ops!($id, $elem_ty);
                test_bitwise_reductions!($id, !(0 as $elem_ty));
                test_all_scalar_shift_ops!($id, $elem_ty);
                test_vector_shift_ops!($id, $elem_ty);
                test_hex_fmt!($id, $elem_ty);
                test_partial_eq!($id, 1 as $elem_ty, 0 as $elem_ty);
                test_default!($id, $elem_ty);
                test_mask_select!($mask_ty, $id, $elem_ty);
                test_int_minmax_ops!($id, $elem_ty);
            }
        );
    }
}

/// Defines a portable packed SIMD mask type.
macro_rules! simd_m_ty {
    ($id:ident : $elem_count:expr, $elem_ty:ident, $test_mod:ident, $test_macro:ident |
     $($elem_tys:ident),+ | $($elem_name:ident),+ | $(#[$doc:meta])*) => {
        vector_impl!(
            [define_ty, $id, $($elem_tys),+ | $(#[$doc])*],
            [impl_mask_minimal, $id, $elem_ty, $elem_count, $($elem_name),*],
            [impl_bitwise_ops, $id, true],
            [impl_bitwise_scalar_ops, $id, bool],
            [impl_mask_bitwise_reductions, $id, bool, $elem_ty],
            [impl_mask_reductions, $id],
            [impl_mask_select, $id, $elem_ty, $elem_count],
            [impl_mask_cmp, $id, $id],
            [impl_eq, $id],
            [impl_partial_eq, $id],
            [impl_default, $id, bool]
        );

        $test_macro!(
            #[cfg(test)]
            mod $test_mod {
                test_mask_minimal!($id, $elem_count);
                test_mask_bitwise_ops!($id);
                test_mask_bitwise_scalar_ops!($id);
                test_mask_reductions!($id);
                test_bitwise_reductions!($id, true);
                test_cmp!($id, $elem_ty, $id, true, false);
                test_partial_eq!($id, true, false);
                test_default!($id, bool);
            }
        );
    }
}
