//! This module implements the [WebAssembly `SIMD128` ISA].
//!
//! [WebAssembly `SIMD128` ISA]:
//! https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md
//
// This files is structured as follows:
// * first the types are defined
// * then macros implementing the different APIs are provided
// * finally the API of each type is implements
//
#![allow(non_camel_case_types)]

#[cfg(test)]
use stdsimd_test::assert_instr;
#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test;

////////////////////////////////////////////////////////////////////////////////
// Types

/// A single unconstrained byte (0-255).
pub type ImmByte = u8;
/// A byte with values in the range 0–1 identifying a lane.
pub type LaneIdx2 = u8;
/// A byte with values in the range 0–3 identifying a lane.
pub type LaneIdx4 = u8;
/// A byte with values in the range 0–7 identifying a lane.
pub type LaneIdx8 = u8;
/// A byte with values in the range 0–15 identifying a lane.
pub type LaneIdx16 = u8;
/// A byte with values in the range 0–31 identifying a lane.
pub type LaneIdx32 = u8;

types! {
    /// WASM-specific 128-bit wide SIMD vector type
    pub struct v128(i128);
}

mod sealed {
    types! {
        /// 128-bit wide SIMD vector type with 8 16-bit wide signed lanes
        pub struct v8x16(
            pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
            pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
        );
        /// 128-bit wide SIMD vector type with 8 16-bit wide signed lanes
        pub struct v16x8(
            pub i16, pub i16, pub i16, pub i16,
            pub i16, pub i16, pub i16, pub i16
        );
        /// 128-bit wide SIMD vector type with 4 32-bit wide signed lanes
        pub struct v32x4(pub i32, pub i32, pub i32, pub i32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide signed lanes
        pub struct v64x2(pub i64, pub i64);

        /// 128-bit wide SIMD vector type with 8 16-bit wide unsigned lanes
        pub struct u8x16(
            pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
            pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
        );
        /// 128-bit wide SIMD vector type with 8 16-bit wide unsigned lanes
        pub struct u16x8(
            pub u16, pub u16, pub u16, pub u16,
            pub u16, pub u16, pub u16, pub u16
        );
        /// 128-bit wide SIMD vector type with 4 32-bit wide unsigned lanes
        pub struct u32x4(pub u32, pub u32, pub u32, pub u32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide unsigned lanes
        pub struct u64x2(pub u64, pub u64);

        /// 128-bit wide SIMD vector type with 4 32-bit wide floating-point lanes
        pub struct f32x4(pub f32, pub f32, pub f32, pub f32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide floating-point lanes
        pub struct f64x2(pub f64, pub f64);
    }

    #[allow(improper_ctypes)]
    extern "C" {
        #[link_name = "llvm.fabs.v4f32"]
        fn abs_v4f32(x: f32x4) -> f32x4;
        #[link_name = "llvm.fabs.v2f64"]
        fn abs_v2f64(x: f64x2) -> f64x2;
        #[link_name = "llvm.sqrt.v4f32"]
        fn sqrt_v4f32(x: f32x4) -> f32x4;
        #[link_name = "llvm.sqrt.v2f64"]
        fn sqrt_v2f64(x: f64x2) -> f64x2;
        #[link_name = "shufflevector"]
        pub fn shufflevector_v16i8(x: v8x16, y: v8x16, i: v8x16) -> v8x16;

    }
    impl f32x4 {
        #[inline(always)]
        pub unsafe fn abs(self) -> Self {
            abs_v4f32(self)
        }
        #[inline(always)]
        pub unsafe fn sqrt(self) -> Self {
            sqrt_v4f32(self)
        }
    }
    impl f64x2 {
        #[inline(always)]
        pub unsafe fn abs(self) -> Self {
            abs_v2f64(self)
        }
        #[inline(always)]
        pub unsafe fn sqrt(self) -> Self {
            sqrt_v2f64(self)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Macros implementing the spec APIs:

macro_rules! impl_splat {
    ($id:ident[$ivec_ty:ident : $elem_ty:ident] <= $x_ty:ident | $($lane_id:ident),*) => {
        /// Create vector with identical lanes
        ///
        /// Construct a vector with `x` replicated to all lanes.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($ident.splat))]
        pub const unsafe fn splat(x: $x_ty) -> v128 {
            union U {
                vec: self::sealed::$ivec_ty,
                res: v128
            }
            U { vec: self::sealed::$ivec_ty($({ struct $lane_id; x as $elem_ty}),*) }.res
        }
    }
}

macro_rules! impl_extract_lane {
    ($id:ident[$ivec_ty:ident : $selem_ty:ident|$uelem_ty:ident]($lane_idx:ty)
     => $x_ty:ident) => {
        /// Extract lane as a scalar (sign-extend)
        ///
        /// Extract the scalar value of lane specified in the immediate
        /// mode operand `imm` from `a` by sign-extending it.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_s, imm =
        // 0))]
        #[rustc_args_required_const(1)]
        pub unsafe fn extract_lane_s(a: v128, imm: $lane_idx) -> $x_ty {
            use coresimd::simd_llvm::simd_extract;
            union U {
                vec: self::sealed::$ivec_ty,
                a: v128,
            }
            // the vectors store a signed integer => extract into it
            let v: $selem_ty = simd_extract(U { a }.vec, imm as u32 /* zero-extends index */);
            v as $x_ty
        }

        /// Extract lane as a scalar (zero-extend)
        ///
        /// Extract the scalar value of lane specified in the immediate
        /// mode operand `imm` from `a` by zero-extending it.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u, imm =
        // 0))]
        #[rustc_args_required_const(1)]
        pub unsafe fn extract_lane_u(a: v128, imm: $lane_idx) -> $x_ty {
            use coresimd::simd_llvm::simd_extract;
            union U {
                vec: self::sealed::$ivec_ty,
                a: v128,
            }
            // the vectors store a signed integer => extract into it
            let v: $selem_ty = simd_extract(U { a }.vec, imm as u32 /* zero-extends index */);
            // re-interpret the signed integer as an unsigned one of the
            // same size (no-op)
            let v: $uelem_ty = ::mem::transmute(v);
            // cast the internal unsigned integer to a larger signed
            // integer (zero-extends)
            v as $x_ty
        }
    };
    ($id:ident[$ivec_ty:ident]($lane_idx:ty) => $x_ty:ident) => {
        /// Extract lane as a scalar
        ///
        /// Extract the scalar value of lane specified in the immediate
        /// mode operand `imm` from `a`.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u, imm =
        // 0))]
        #[rustc_args_required_const(1)]
        pub unsafe fn extract_lane(a: v128, imm: $lane_idx) -> $x_ty {
            use coresimd::simd_llvm::simd_extract;
            union U {
                vec: self::sealed::$ivec_ty,
                a: v128,
            }
            // the vectors store a signed integer => extract into it
            simd_extract(U { a }.vec, imm as u32 /* zero-extends index */)
        }
    };
}

macro_rules! impl_replace_lane {
    ($id:ident[$ivec_ty:ident:$ielem_ty:ident]($lane_idx:ty) <= $x_ty:ident) => {
        /// Replace lane value
        ///
        /// Return a new vector with lanes identical to `a`, except for
        /// lane specified in the immediate mode argument `i` which
        /// has the value `x`.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u))]
        #[rustc_args_required_const(1)]
        pub unsafe fn replace_lane(a: v128, imm: $lane_idx, x: $x_ty) -> v128 {
            use coresimd::simd_llvm::simd_insert;
            union U {
                vec: self::sealed::$ivec_ty,
                a: v128,
            }
            // the vectors store a signed integer => extract into it
            ::mem::transmute(simd_insert(
                U { a }.vec,
                imm as u32, /* zero-extends index */
                x as $ielem_ty,
            ))
        }
    };
}

macro_rules! impl_wrapping_add_sub_neg {
    ($id:ident[$ivec_ty:ident]) => {
        /// Lane-wise wrapping integer addition
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.add))]
        pub unsafe fn add(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_add;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            ::mem::transmute(simd_add(a, b))
        }

        /// Lane-wise wrapping integer subtraction
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.sub))]
        pub unsafe fn sub(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_sub;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            ::mem::transmute(simd_sub(a, b))
        }

        /// Lane-wise wrapping integer negation
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.neg))]
        pub unsafe fn neg(a: v128) -> v128 {
            use coresimd::simd_llvm::simd_mul;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute($id::splat(-1));
            ::mem::transmute(simd_mul(b, a))
        }

        // note: multiplication explicitly omitted because i64x2 does
        // not implement it
    };
}

// TODO: Saturating integer arithmetic
// need to add intrinsics to rustc

// note: multiplication explicitly implemented separately because i64x2 does
// not implement it

macro_rules! impl_wrapping_mul {
    ($id:ident[$ivec_ty:ident]) => {
        /// Lane-wise wrapping integer multiplication
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.mul))]
        pub unsafe fn mul(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_mul;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            ::mem::transmute(simd_mul(a, b))
        }
    };
}

macro_rules! impl_shl_scalar {
    ($id:ident[$ivec_ty:ident : $t:ty]) => {
        /// Left shift by scalar.
        ///
        /// Shift the bits in each lane to the left by the same amount.
        /// Only the low bits of the shift amount are used.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.shl))]
        pub unsafe fn shl(a: v128, y: i32) -> v128 {
            use coresimd::simd_llvm::simd_shl;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute($id::splat(y as $t));
            ::mem::transmute(simd_shl(a, b))
        }
    };
}

macro_rules! impl_shr_scalar {
    ($id:ident[$svec_ty:ident : $uvec_ty:ident : $t:ty]) => {
        /// Arithmetic right shift by scalar.
        ///
        /// Shift the bits in each lane to the right by the same amount.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.shr))]
        pub unsafe fn shr_s(a: v128, y: i32) -> v128 {
            use coresimd::simd_llvm::simd_shr;
            let a: sealed::$svec_ty = ::mem::transmute(a);
            let b: sealed::$svec_ty = ::mem::transmute($id::splat(y as $t));
            ::mem::transmute(simd_shr(a, b))
        }

        /// Logical right shift by scalar.
        ///
        /// Shift the bits in each lane to the right by the same amount.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.shr))]
        pub unsafe fn shr_u(a: v128, y: i32) -> v128 {
            use coresimd::simd_llvm::simd_shr;
            let a: sealed::$uvec_ty = ::mem::transmute(a);
            let b: sealed::$uvec_ty = ::mem::transmute($id::splat(y as $t));
            ::mem::transmute(simd_shr(a, b))
        }
    };
}

macro_rules! impl_boolean_reduction {
    ($id:ident[$ivec_ty:ident]) => {
        /// Any lane true
        ///
        /// Returns `1` if any lane in `a` is non-zero, `0` otherwise.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.any_true))]
        pub unsafe fn any_true(a: v128) -> i32 {
            use coresimd::simd_llvm::simd_reduce_any;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            if simd_reduce_any(a) {
                1
            } else {
                0
            }
        }

        /// All lanes true
        ///
        /// Returns `1` if all lanes in `a` are non-zero, `0` otherwise.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.all_true))]
        pub unsafe fn all_true(a: v128) -> i32 {
            use coresimd::simd_llvm::simd_reduce_all;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            if simd_reduce_all(a) {
                1
            } else {
                0
            }
        }
    };
}

macro_rules! impl_comparisons {
    ($id:ident[$ivec_ty:ident]) => {
        impl_comparisons!($id[$ivec_ty=>$ivec_ty]);
    };
    ($id:ident[$ivec_ty:ident=>$rvec_ty:ident]) => {
        /// Equality
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.eq))]
        pub unsafe fn eq(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_eq;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_eq(a, b);
            ::mem::transmute(c)
        }
        /// Non-Equality
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.ne))]
        pub unsafe fn ne(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_ne;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_ne(a, b);
            ::mem::transmute(c)
        }
        /// Less-than
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.lt))]
        pub unsafe fn lt(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_lt;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_lt(a, b);
            ::mem::transmute(c)
        }
        /// Less-than or equal
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.le))]
        pub unsafe fn le(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_le;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_le(a, b);
            ::mem::transmute(c)
        }
        /// Greater-than
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.gt))]
        pub unsafe fn gt(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_gt;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_gt(a, b);
            ::mem::transmute(c)
        }
        /// Greater-than or equal
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.ge))]
        pub unsafe fn ge(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_ge;
            let a: sealed::$ivec_ty = ::mem::transmute(a);
            let b: sealed::$ivec_ty = ::mem::transmute(b);
            let c: sealed::$rvec_ty = simd_ge(a, b);
            ::mem::transmute(c)
        }
    }
}

// Floating-point operations
macro_rules! impl_floating_point_ops {
    ($id:ident) => {
        /// Negation
        ///
        /// Apply the IEEE `negate(x)` function to each lane. This simply
        /// inverts the sign bit, preserving all other bits, even for `NaN`
        /// inputs.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.neg))]
        pub unsafe fn neg(a: v128) -> v128 {
            use coresimd::simd_llvm::simd_mul;
            let a: sealed::$id = ::mem::transmute(a);
            let b: sealed::$id = ::mem::transmute($id::splat(-1.));
            ::mem::transmute(simd_mul(b, a))
        }
        /// Absolute value
        ///
        /// Apply the IEEE `abs(x)` function to each lane. This simply
        /// clears the sign bit, preserving all other bits, even for `NaN`
        /// inputs.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.abs))]
        pub unsafe fn abs(a: v128) -> v128 {
            let a: sealed::$id = ::mem::transmute(a);
            ::mem::transmute(a.abs())
        }
        /// NaN-propagating minimum
        ///
        /// Lane-wise minimum value, propagating `NaN`s.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.min))]
        pub unsafe fn min(a: v128, b: v128) -> v128 {
            v128::bitselect(a, b, $id::lt(a, b))
        }
        /// NaN-propagating maximum
        ///
        /// Lane-wise maximum value, propagating `NaN`s.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.max))]
        pub unsafe fn max(a: v128, b: v128) -> v128 {
            v128::bitselect(a, b, $id::gt(a, b))
        }
        /// Square-root
        ///
        /// Lane-wise square-root.
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.sqrt))]
        pub unsafe fn sqrt(a: v128) -> v128 {
            let a: sealed::$id = ::mem::transmute(a);
            ::mem::transmute(a.sqrt())
        }
        /// Lane-wise addition
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.add))]
        pub unsafe fn add(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_add;
            let a: sealed::$id = ::mem::transmute(a);
            let b: sealed::$id = ::mem::transmute(b);
            ::mem::transmute(simd_add(a, b))
        }
        /// Lane-wise subtraction
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.sub))]
        pub unsafe fn sub(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_sub;
            let a: sealed::$id = ::mem::transmute(a);
            let b: sealed::$id = ::mem::transmute(b);
            ::mem::transmute(simd_sub(a, b))
        }
        /// Lane-wise multiplication
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.mul))]
        pub unsafe fn mul(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_mul;
            let a: sealed::$id = ::mem::transmute(a);
            let b: sealed::$id = ::mem::transmute(b);
            ::mem::transmute(simd_mul(a, b))
        }
        /// Lane-wise division
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($id.div))]
        pub unsafe fn div(a: v128, b: v128) -> v128 {
            use coresimd::simd_llvm::simd_div;
            let a: sealed::$id = ::mem::transmute(a);
            let b: sealed::$id = ::mem::transmute(b);
            ::mem::transmute(simd_div(a, b))
        }
    };
}

macro_rules! impl_conversion {
    ($conversion:ident[$instr:expr]: $from_ty:ident => $to_ty:ident | $id:ident) => {
        #[inline]
        // #[target_feature(enable = "simd128")]
        // FIXME: #[cfg_attr(test, assert_instr($instr))]
        pub unsafe fn $conversion(a: v128) -> v128 {
            use coresimd::simd_llvm::simd_cast;
            let a: sealed::$from_ty = ::mem::transmute(a);
            let b: sealed::$to_ty = simd_cast(a);
            ::mem::transmute(b)
        }
    };
}

////////////////////////////////////////////////////////////////////////////////
// Implementations:

// v128
impl v128 {
    ///////////////////////////////////////////////////////////////////////////
    // Const constructor:

    /// Materialize a constant SIMD value from the immediate operands.
    ///
    /// The `v128.const` instruction is encoded with 16 immediate bytes
    /// `imm` which provide the bits of the vector directly.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr(v128.const, imm =
    // [ImmByte::new(42); 16]))]
    #[rustc_args_required_const(0)]
    pub const unsafe fn const_(imm: [ImmByte; 16]) -> v128 {
        union U {
            imm: [ImmByte; 16],
            vec: v128,
        }
        U { imm }.vec
    }

    ///////////////////////////////////////////////////////////////////////////
    // Bitwise logical operations:

    /// Bitwise logical and
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.and))]
    pub unsafe fn and(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_and;
        simd_and(a, b)
    }

    /// Bitwise logical or
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.or))]
    pub unsafe fn or(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_or;
        simd_or(a, b)
    }

    /// Bitwise logical xor
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.xor))]
    pub unsafe fn xor(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_xor;
        simd_xor(a, b)
    }

    /// Bitwise logical not
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.not))]
    pub unsafe fn not(a: v128) -> v128 {
        union U {
            v: u128,
            c: [ImmByte; 16],
        }
        // FIXME: https://github.com/rust-lang/rust/issues/53193
        const C: [ImmByte; 16] = unsafe { U { v: ::u128::MAX }.c };
        Self::xor(v128::const_(C), a)
    }

    /// Bitwise select
    ///
    /// Use the bits in the control mask `c` to select the corresponding bit
    /// from `v1` when `1` and `v2` when `0`.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.bitselect))]
    pub unsafe fn bitselect(v1: v128, v2: v128, c: v128) -> v128 {
        // FIXME: use llvm.select instead - we need to add a `simd_bitselect`
        // intrinsic to rustc that converts a v128 vector into a i1x128. The
        // `simd_select` intrinsic converts e.g. a i8x16 into a i1x16 which is
        // not what we want here:
        Self::or(Self::and(v1, c), Self::and(v2, Self::not(c)))
    }

    ///////////////////////////////////////////////////////////////////////////
    // Memory load/stores:

    /// Load a `v128` vector from the given heap address.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.load))]
    pub unsafe fn load(m: *const v128) -> v128 {
        ::ptr::read(m)
    }

    /// Store a `v128` vector to the given heap address.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.store))]
    pub unsafe fn store(m: *mut v128, a: v128) {
        ::ptr::write(m, a)
    }
}

pub use self::sealed::v8x16 as __internal_v8x16;
pub use coresimd::simd_llvm::simd_shuffle16 as __internal_v8x16_shuffle;
/// Shuffle lanes
///
/// Create vector with lanes selected from the lanes of two input vectors
/// `a` and `b` by the indices specified in the immediate mode operand
/// `imm`. Each index selects an element of the result vector, where the
/// indices `i` in range `[0, 15]` select the `i`-th elements of `a`, and
/// the indices in range `[16, 31]` select the `i - 16`-th element of `b`.
#[macro_export]
macro_rules! v8x16_shuffle {
    ($a:expr, $b:expr, [
        $imm0:expr, $imm1:expr, $imm2:expr, $imm3:expr,
        $imm4:expr, $imm5:expr, $imm6:expr, $imm7:expr,
        $imm8:expr, $imm9:expr, $imm10:expr, $imm11:expr,
        $imm12:expr, $imm13:expr, $imm14:expr, $imm15:expr
    ]) => {
        #[allow(unused_unsafe)]
        unsafe {
            let a: $crate::arch::wasm32::v128 = $a;
            let b: $crate::arch::wasm32::v128 = $b;
            union U {
                e: v128,
                i: $crate::arch::wasm32::__internal_v8x16,
            }
            let a = U { e: a }.i;
            let b = U { e: b }.i;

            let r: $crate::arch::wasm32::__internal_v8x16 =
                $crate::arch::wasm32::__internal_v8x16_shuffle(
                    a,
                    b,
                    [
                        $imm0 as u32,
                        $imm1,
                        $imm2,
                        $imm3,
                        $imm4,
                        $imm5,
                        $imm6,
                        $imm7,
                        $imm8,
                        $imm9,
                        $imm10,
                        $imm11,
                        $imm12,
                        $imm13,
                        $imm14,
                        $imm15,
                    ],
                );
            U { i: r }.e
        }
    };
}

/// WASM-specific v8x16 instructions with modulo-arithmetic semantics
pub mod i8x16 {
    use super::*;
    impl_splat!(
        i8x16[v8x16: i8] <= i32 | x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15
    );
    impl_extract_lane!(i8x16[v8x16:i8|u8](LaneIdx16) => i32);
    impl_replace_lane!(i8x16[v8x16: i8](LaneIdx16) <= i32);
    impl_wrapping_add_sub_neg!(i8x16[v8x16]);
    impl_wrapping_mul!(i8x16[v8x16]);
    impl_shl_scalar!(i8x16[v8x16: i32]);
    impl_shr_scalar!(i8x16[v8x16: u8x16: i32]);
    impl_boolean_reduction!(i8x16[v8x16]);
    impl_comparisons!(i8x16[v8x16]);
}

/// WASM-specific v16x8 instructions with modulo-arithmetic semantics
pub mod i16x8 {
    use super::*;
    impl_splat!(i16x8[v16x8: i16] <= i32 | x0, x1, x2, x3, x4, x5, x6, x7);
    impl_extract_lane!(i16x8[v16x8:i16|u16](LaneIdx8) => i32);
    impl_replace_lane!(i16x8[v16x8: i16](LaneIdx8) <= i32);
    impl_wrapping_add_sub_neg!(i16x8[v16x8]);
    impl_wrapping_mul!(i16x8[v16x8]);
    impl_shl_scalar!(i16x8[v16x8: i32]);
    impl_shr_scalar!(i16x8[v16x8: u16x8: i32]);
    impl_boolean_reduction!(i16x8[v16x8]);
    impl_comparisons!(i16x8[v16x8]);
}

/// WASM-specific v32x4 instructions with modulo-arithmetic semantics
pub mod i32x4 {
    use super::*;
    impl_splat!(i32x4[v32x4: i32] <= i32 | x0, x1, x2, x3);
    impl_extract_lane!(i32x4[v32x4](LaneIdx4) => i32);
    impl_replace_lane!(i32x4[v32x4: i32](LaneIdx4) <= i32);
    impl_wrapping_add_sub_neg!(i32x4[v32x4]);
    impl_wrapping_mul!(i32x4[v32x4]);
    impl_shl_scalar!(i32x4[v32x4: i32]);
    impl_shr_scalar!(i32x4[v32x4: u32x4: i32]);
    impl_boolean_reduction!(i32x4[v32x4]);
    impl_comparisons!(i32x4[v32x4]);
    impl_conversion!(trunc_s_f32x4_sat["i32x4.trunc_s/f32x4:sat"]: f32x4 => v32x4 | i32x4);
    impl_conversion!(trunc_u_f32x4_sat["i32x4.trunc_s/f32x4:sat"]: f32x4 => u32x4 | i32x4);
}

/// WASM-specific v64x2 instructions with modulo-arithmetic semantics
pub mod i64x2 {
    use super::*;
    impl_splat!(i64x2[v64x2: i64] <= i64 | x0, x1);
    impl_extract_lane!(i64x2[v64x2](LaneIdx2) => i64);
    impl_replace_lane!(i64x2[v64x2: i64](LaneIdx2) <= i64);
    impl_wrapping_add_sub_neg!(i64x2[v64x2]);
    // note: wrapping multiplication for i64x2 is not part of the spec
    impl_shl_scalar!(i64x2[v64x2: i64]);
    impl_shr_scalar!(i64x2[v64x2: u64x2: i64]);
    impl_boolean_reduction!(i64x2[v64x2]);
    impl_comparisons!(i64x2[v64x2]);
    impl_conversion!(trunc_s_f64x2_sat["i64x2.trunc_s/f64x2:sat"]: f64x2 => v64x2 | i64x2);
    impl_conversion!(trunc_u_f64x2_sat["i64x2.trunc_s/f64x2:sat"]: f64x2 => u64x2 | i64x2);
}

/// WASM-specific v32x4 floating-point instructions
pub mod f32x4 {
    use super::*;
    impl_splat!(f32x4[f32x4: f32] <= f32 | x0, x1, x2, x3);
    impl_extract_lane!(f32x4[f32x4](LaneIdx4) => f32);
    impl_replace_lane!(f32x4[f32x4: f32](LaneIdx4) <= f32);
    impl_comparisons!(f32x4[f32x4=>v32x4]);
    impl_floating_point_ops!(f32x4);
    impl_conversion!(convert_s_i32x4["f32x4.convert_s/i32x4"]: v32x4 => f32x4 | f32x4);
    impl_conversion!(convert_u_i32x4["f32x4.convert_u/i32x4"]: u32x4 => f32x4 | f32x4);

}

/// WASM-specific v64x2 floating-point instructions
pub mod f64x2 {
    use super::*;
    impl_splat!(f64x2[f64x2: f64] <= f64 | x0, x1);
    impl_extract_lane!(f64x2[f64x2](LaneIdx2) => f64);
    impl_replace_lane!(f64x2[f64x2: f64](LaneIdx2) <= f64);
    impl_comparisons!(f64x2[f64x2=>v64x2]);
    impl_floating_point_ops!(f64x2);
    impl_conversion!(convert_s_i64x2["f64x2.convert_s/i64x2"]: v64x2 => f64x2 | f64x2);
    impl_conversion!(convert_u_i64x2["f64x2.convert_u/i64x2"]: u64x2 => f64x2 | f64x2);
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std;
    use std::mem;
    use std::prelude::v1::*;
    use wasm_bindgen_test::*;

    fn compare_bytes(a: v128, b: v128) {
        let a: [u8; 16] = unsafe { mem::transmute(a) };
        let b: [u8; 16] = unsafe { mem::transmute(b) };
        assert_eq!(a, b);
    }

    #[wasm_bindgen_test]
    fn v128_const() {
        const A: v128 =
            unsafe { v128::const_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) };
        compare_bytes(A, A);
    }

    macro_rules! test_splat {
        ($test_id:ident: $id:ident($val:expr) => $($vals:expr),*) => {
            #[wasm_bindgen_test]
            fn $test_id() {
                const A: v128 = unsafe {
                    $id::splat($val)
                };
                const B: v128 = unsafe {
                    v128::const_([$($vals),*])
                };
                compare_bytes(A, B);
            }
        }
    }

    test_splat!(i8x16_splat: i8x16(42) => 42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42);
    test_splat!(i16x8_splat: i16x8(42) => 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0);
    test_splat!(i32x4_splat: i32x4(42) => 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0);
    test_splat!(i64x2_splat: i64x2(42) => 42, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0);
    test_splat!(f32x4_splat: f32x4(42.) => 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66);
    test_splat!(f64x2_splat: f64x2(42.) => 0, 0, 0, 0, 0, 0, 69, 64, 0, 0, 0, 0, 0, 0, 69, 64);

    // tests extract and replace lanes
    macro_rules! test_extract {
        ($test_id:ident: $id:ident[$ety:ident] => $extract_fn:ident | [$val:expr; $count:expr]
         | [$($vals:expr),*] => ($other:expr)
         | $($ids:expr),*) => {
            #[wasm_bindgen_test]
            fn $test_id() {
                unsafe {
                    // splat vector and check that all indices contain the same value
                    // splatted:
                    const A: v128 = unsafe {
                        $id::splat($val)
                    };
                    $(
                        assert_eq!($id::$extract_fn(A, $ids) as $ety, $val);
                    )*;

                    // create a vector from array and check that the indices contain
                    // the same values as in the array:
                    let arr: [$ety; $count] = [$($vals),*];
                    let mut vec: v128 = mem::transmute(arr);
                    $(
                        assert_eq!($id::$extract_fn(vec, $ids) as $ety, arr[$ids]);
                    )*;

                    // replace lane 0 with another value
                    vec = $id::replace_lane(vec, 0, $other);
                    assert_ne!($id::$extract_fn(vec, 0) as $ety, arr[0]);
                    assert_eq!($id::$extract_fn(vec, 0) as $ety, $other);
                }
            }
        }
    }

    test_extract!(i8x16_extract_u: i8x16[u8] => extract_lane_u | [255; 16]
                  | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] => (42)
                  | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    test_extract!(i8x16_extract_s: i8x16[i8] => extract_lane_s | [-122; 16]
                  | [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15] => (-42)
                  | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );

    test_extract!(i16x8_extract_u: i16x8[u16] => extract_lane_u | [255; 8]
                  | [0, 1, 2, 3, 4, 5, 6, 7]  => (42) | 0, 1, 2, 3, 4, 5, 6, 7
    );
    test_extract!(i16x8_extract_s: i16x8[i16] => extract_lane_s | [-122; 8]
                  | [0, -1, 2, -3, 4, -5, 6, -7]  => (-42) | 0, 1, 2, 3, 4, 5, 6, 7
    );
    test_extract!(i32x4_extract: i32x4[i32] => extract_lane | [-122; 4]
                  | [0, -1, 2, -3]  => (42) | 0, 1, 2, 3
    );
    test_extract!(i64x2_extract: i64x2[i64] => extract_lane | [-122; 2]
                  | [0, -1]  => (42) | 0, 1
    );
    test_extract!(f32x4_extract: f32x4[f32] => extract_lane | [-122.; 4]
                  | [0., -1., 2., -3.]  => (42.) | 0, 1, 2, 3
    );
    test_extract!(f64x2_extract: f64x2[f64] => extract_lane | [-122.; 2]
                  | [0., -1.]  => (42.) | 0, 1
    );

    #[wasm_bindgen_test]
    fn v8x16_shuffle() {
        unsafe {
            let a = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b = [
                16_u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];

            let vec_a: v128 = mem::transmute(a);
            let vec_b: v128 = mem::transmute(b);

            let vec_r = v8x16_shuffle!(
                vec_a,
                vec_b,
                [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
            );

            let e = [0_u8, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
            let vec_e: v128 = mem::transmute(e);
            compare_bytes(vec_r, vec_e);
        }
    }

    macro_rules! floating_point {
        (f32) => {
            true
        };
        (f64) => {
            true
        };
        ($id:ident) => {
            false
        };
    }

    trait IsNan: Sized {
        fn is_nan(self) -> bool {
            false
        }
    }
    impl IsNan for i8 {}
    impl IsNan for i16 {}
    impl IsNan for i32 {}
    impl IsNan for i64 {}

    macro_rules! test_bop {
        ($id:ident[$ety:ident; $ecount:expr] |
         $binary_op:ident [$op_test_id:ident] :
         ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
            test_bop!(
                $id[$ety; $ecount] => $ety | $binary_op [ $op_test_id ]:
                ([$($in_a),*], [$($in_b),*]) => [$($out),*]
            );

        };
        ($id:ident[$ety:ident; $ecount:expr] => $oty:ident |
         $binary_op:ident [$op_test_id:ident] :
         ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
            #[wasm_bindgen_test]
            fn $op_test_id() {
                unsafe {
                    let a_input: [$ety; $ecount] = [$($in_a),*];
                    let b_input: [$ety; $ecount] = [$($in_b),*];
                    let output: [$oty; $ecount] = [$($out),*];

                    let a_vec_in: v128 = mem::transmute(a_input);
                    let b_vec_in: v128 = mem::transmute(b_input);
                    let vec_res: v128 = $id::$binary_op(a_vec_in, b_vec_in);

                    let res: [$oty; $ecount] = mem::transmute(vec_res);

                    if !floating_point!($ety) {
                        assert_eq!(res, output);
                    } else {
                        for i in 0..$ecount {
                            let r = res[i];
                            let o = output[i];
                            assert_eq!(r.is_nan(), o.is_nan());
                            if !r.is_nan() {
                                assert_eq!(r, o);
                            }
                        }
                    }
                }
            }
        }
    }

    macro_rules! test_bops {
        ($id:ident[$ety:ident; $ecount:expr] |
         $binary_op:ident [$op_test_id:ident]:
         ([$($in_a:expr),*], $in_b:expr) => [$($out:expr),*]) => {
            #[wasm_bindgen_test]
            fn $op_test_id() {
                unsafe {
                    let a_input: [$ety; $ecount] = [$($in_a),*];
                    let output: [$ety; $ecount] = [$($out),*];

                    let a_vec_in: v128 = mem::transmute(a_input);
                    let vec_res: v128 = $id::$binary_op(a_vec_in, $in_b);

                    let res: [$ety; $ecount] = mem::transmute(vec_res);
                    assert_eq!(res, output);
                }
            }
        }
    }

    macro_rules! test_uop {
        ($id:ident[$ety:ident; $ecount:expr] |
         $unary_op:ident [$op_test_id:ident]: [$($in_a:expr),*] => [$($out:expr),*]) => {
            #[wasm_bindgen_test]
            fn $op_test_id() {
                unsafe {
                    let a_input: [$ety; $ecount] = [$($in_a),*];
                    let output: [$ety; $ecount] = [$($out),*];

                    let a_vec_in: v128 = mem::transmute(a_input);
                    let vec_res: v128 = $id::$unary_op(a_vec_in);

                    let res: [$ety; $ecount] = mem::transmute(vec_res);
                    assert_eq!(res, output);
                }
            }
        }
    }

    test_bop!(i8x16[i8; 16] | add[i8x16_add_test]:
              ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1],
               [8, i8::min_value(), 10, 11, 12, 13, 14, 1, 1, 1, 1, 1, 1, 1, 1, 1]) =>
              [8, i8::max_value(), 12, 14, 16, 18, 20, i8::min_value(), 2, 2, 2, 2, 2, 2, 2, 2]);
    test_bop!(i8x16[i8; 16] | sub[i8x16_sub_test]:
              ([0, -1, 2, 3, 4, 5, 6, -1, 1, 1, 1, 1, 1, 1, 1, 1],
               [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
              [-8, i8::max_value(), -8, -8, -8, -8, -8, i8::min_value(), 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bop!(i8x16[i8; 16] | mul[i8x16_mul_test]:
              ([0, -2, 2, 3, 4, 5, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1],
               [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
              [0, 0, 20, 33, 48, 65, 84, -2, 1, 1, 1, 1, 1, 1, 1, 1]);
    test_uop!(i8x16[i8; 16] | neg[i8x16_neg_test]:
              [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1] =>
              [-8, i8::min_value(), -10, -11, -12, -13, -14, i8::min_value() + 1, -1, -1, -1, -1, -1, -1, -1, -1]);

    test_bop!(i16x8[i16; 8] | add[i16x8_add_test]:
              ([0, -1, 2, 3, 4, 5, 6, i16::max_value()],
               [8, i16::min_value(), 10, 11, 12, 13, 14, 1]) =>
              [8, i16::max_value(), 12, 14, 16, 18, 20, i16::min_value()]);
    test_bop!(i16x8[i16; 8] | sub[i16x8_sub_test]:
              ([0, -1, 2, 3, 4, 5, 6, -1],
               [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
              [-8, i16::max_value(), -8, -8, -8, -8, -8, i16::min_value()]);
    test_bop!(i16x8[i16; 8] | mul[i16x8_mul_test]:
              ([0, -2, 2, 3, 4, 5, 6, 2],
               [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
              [0, 0, 20, 33, 48, 65, 84, -2]);
    test_uop!(i16x8[i16; 8] | neg[i16x8_neg_test]:
              [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()] =>
              [-8, i16::min_value(), -10, -11, -12, -13, -14, i16::min_value() + 1]);

    test_bop!(i32x4[i32; 4] | add[i32x4_add_test]:
              ([0, -1, 2, i32::max_value()],
               [8, i32::min_value(), 10, 1]) =>
              [8, i32::max_value(), 12, i32::min_value()]);
    test_bop!(i32x4[i32; 4] | sub[i32x4_sub_test]:
              ([0, -1, 2, -1],
               [8, i32::min_value(), 10, i32::max_value()]) =>
              [-8, i32::max_value(), -8, i32::min_value()]);
    test_bop!(i32x4[i32; 4] | mul[i32x4_mul_test]:
              ([0, -2, 2, 2],
               [8, i32::min_value(), 10, i32::max_value()]) =>
              [0, 0, 20, -2]);
    test_uop!(i32x4[i32; 4] | neg[i32x4_neg_test]:
              [8, i32::min_value(), 10, i32::max_value()] =>
              [-8, i32::min_value(), -10, i32::min_value() + 1]);

    test_bop!(i64x2[i64; 2] | add[i64x2_add_test]:
              ([-1, i64::max_value()],
               [i64::min_value(), 1]) =>
              [i64::max_value(), i64::min_value()]);
    test_bop!(i64x2[i64; 2] | sub[i64x2_sub_test]:
              ([-1, -1],
               [i64::min_value(), i64::max_value()]) =>
              [ i64::max_value(), i64::min_value()]);
    // note: mul for i64x2 is not part of the spec
    test_uop!(i64x2[i64; 2] | neg[i64x2_neg_test]:
              [i64::min_value(), i64::max_value()] =>
              [i64::min_value(), i64::min_value() + 1]);

    test_bops!(i8x16[i8; 16] | shl[i8x16_shl_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -2, 4, 6, 8, 10, 12, -2, 2, 2, 2, 2, 2, 2, 2, 2]);
    test_bops!(i16x8[i16; 8] | shl[i16x8_shl_test]:
               ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
               [0, -2, 4, 6, 8, 10, 12, -2]);
    test_bops!(i32x4[i32; 4] | shl[i32x4_shl_test]:
               ([0, -1, 2, 3], 1) => [0, -2, 4, 6]);
    test_bops!(i64x2[i64; 2] | shl[i64x2_shl_test]:
               ([0, -1], 1) => [0, -2]);

    test_bops!(i8x16[i8; 16] | shr_s[i8x16_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, -1, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | shr_s[i16x8_shr_s_test]:
               ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
               [0, -1, 1, 1, 2, 2, 3, i16::max_value() / 2]);
    test_bops!(i32x4[i32; 4] | shr_s[i32x4_shr_s_test]:
               ([0, -1, 2, 3], 1) => [0, -1, 1, 1]);
    test_bops!(i64x2[i64; 2] | shr_s[i64x2_shr_s_test]:
               ([0, -1], 1) => [0, -1]);

    test_bops!(i8x16[i8; 16] | shr_u[i8x16_uhr_u_test]:
               ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
               [0, i8::max_value(), 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
    test_bops!(i16x8[i16; 8] | shr_u[i16x8_uhr_u_test]:
               ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
               [0, i16::max_value(), 1, 1, 2, 2, 3, i16::max_value() / 2]);
    test_bops!(i32x4[i32; 4] | shr_u[i32x4_uhr_u_test]:
               ([0, -1, 2, 3], 1) => [0, i32::max_value(), 1, 1]);
    test_bops!(i64x2[i64; 2] | shr_u[i64x2_uhr_u_test]:
               ([0, -1], 1) => [0, i64::max_value()]);

    #[wasm_bindgen_test]
    fn v128_bitwise_logical_ops() {
        unsafe {
            let a: [u32; 4] = [u32::max_value(), 0, u32::max_value(), 0];
            let b: [u32; 4] = [u32::max_value(); 4];
            let c: [u32; 4] = [0; 4];

            let vec_a: v128 = mem::transmute(a);
            let vec_b: v128 = mem::transmute(b);
            let vec_c: v128 = mem::transmute(c);

            let r: v128 = v128::and(vec_a, vec_a);
            compare_bytes(r, vec_a);
            let r: v128 = v128::and(vec_a, vec_b);
            compare_bytes(r, vec_a);
            let r: v128 = v128::or(vec_a, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128::not(vec_b);
            compare_bytes(r, vec_c);
            let r: v128 = v128::xor(vec_a, vec_c);
            compare_bytes(r, vec_a);

            let r: v128 = v128::bitselect(vec_b, vec_c, vec_b);
            compare_bytes(r, vec_b);
            let r: v128 = v128::bitselect(vec_b, vec_c, vec_c);
            compare_bytes(r, vec_c);
            let r: v128 = v128::bitselect(vec_b, vec_c, vec_a);
            compare_bytes(r, vec_a);
        }
    }

    macro_rules! test_bool_red {
        ($id:ident[$test_id:ident] | [$($true:expr),*] | [$($false:expr),*] | [$($alt:expr),*]) => {
            #[wasm_bindgen_test]
            fn $test_id() {
                unsafe {
                    let vec_a: v128 = mem::transmute([$($true),*]); // true
                    let vec_b: v128 = mem::transmute([$($false),*]); // false
                    let vec_c: v128 = mem::transmute([$($alt),*]); // alternating

                    assert_eq!($id::any_true(vec_a), 1);
                    assert_eq!($id::any_true(vec_b), 0);
                    assert_eq!($id::any_true(vec_c), 1);

                    assert_eq!($id::all_true(vec_a), 1);
                    assert_eq!($id::all_true(vec_b), 0);
                    assert_eq!($id::all_true(vec_c), 0);
                }
            }
        }
    }

    test_bool_red!(
        i8x16[i8x16_boolean_reductions]
            | [1_i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            | [0_i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            | [1_i8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        i16x8[i16x8_boolean_reductions]
            | [1_i16, 1, 1, 1, 1, 1, 1, 1]
            | [0_i16, 0, 0, 0, 0, 0, 0, 0]
            | [1_i16, 0, 1, 0, 1, 0, 1, 0]
    );
    test_bool_red!(
        i32x4[i32x4_boolean_reductions] | [1_i32, 1, 1, 1] | [0_i32, 0, 0, 0] | [1_i32, 0, 1, 0]
    );
    test_bool_red!(i64x2[i64x2_boolean_reductions] | [1_i64, 1] | [0_i64, 0] | [1_i64, 0]);

    test_bop!(i8x16[i8; 16] | eq[i8x16_eq_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | eq[i16x8_eq_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | eq[i32x4_eq_test]:
              ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(i64x2[i64; 2] | eq[i64x2_eq_test]: ([0, 1], [0, 2]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | eq[f32x4_eq_test]:
              ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | eq[f64x2_eq_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | ne[i8x16_ne_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | ne[i16x8_ne_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | ne[i32x4_ne_test]:
              ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | ne[i64x2_ne_test]: ([0, 1], [0, 2]) => [0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | ne[f32x4_ne_test]:
              ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | ne[f64x2_ne_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | lt[i8x16_lt_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | lt[i16x8_lt_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | lt[i32x4_lt_test]:
              ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | lt[i64x2_lt_test]: ([0, 1], [0, 2]) => [0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | lt[f32x4_lt_test]:
              ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | lt[f64x2_lt_test]: ([0., 1.], [0., 2.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | gt[i8x16_gt_test]:
          ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i16x8[i16; 8] | gt[i16x8_gt_test]:
              ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
              [0, -1, 0, -1 ,0, -1, 0, 0]);
    test_bop!(i32x4[i32; 4] | gt[i32x4_gt_test]:
              ([0, 2, 2, 4], [0, 1, 2, 3]) => [0, -1, 0, -1]);
    test_bop!(i64x2[i64; 2] | gt[i64x2_gt_test]: ([0, 2], [0, 1]) => [0, -1]);
    test_bop!(f32x4[f32; 4] => i32 | gt[f32x4_gt_test]:
              ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [0, -1, 0, -1]);
    test_bop!(f64x2[f64; 2] => i64 | gt[f64x2_gt_test]: ([0., 2.], [0., 1.]) => [0, -1]);

    test_bop!(i8x16[i8; 16] | ge[i8x16_ge_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | ge[i16x8_ge_test]:
              ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | ge[i32x4_ge_test]:
              ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
    test_bop!(i64x2[i64; 2] | ge[i64x2_ge_test]: ([0, 1], [0, 2]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | ge[f32x4_ge_test]:
              ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
    test_bop!(f64x2[f64; 2] => i64 | ge[f64x2_ge_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

    test_bop!(i8x16[i8; 16] | le[i8x16_le_test]:
              ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
              ) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i16x8[i16; 8] | le[i16x8_le_test]:
              ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
              [-1, 0, -1, 0 ,-1, 0, -1, -1]);
    test_bop!(i32x4[i32; 4] | le[i32x4_le_test]:
              ([0, 2, 2, 4], [0, 1, 2, 3]) => [-1, 0, -1, 0]);
    test_bop!(i64x2[i64; 2] | le[i64x2_le_test]: ([0, 2], [0, 1]) => [-1, 0]);
    test_bop!(f32x4[f32; 4] => i32 | le[f32x4_le_test]:
              ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [-1, 0, -1, -0]);
    test_bop!(f64x2[f64; 2] => i64 | le[f64x2_le_test]: ([0., 2.], [0., 1.]) => [-1, 0]);

    #[wasm_bindgen_test]
    fn v128_bitwise_load_store() {
        unsafe {
            let mut arr: [i32; 4] = [0, 1, 2, 3];

            let vec = v128::load(arr.as_ptr() as *const v128);
            let vec = i32x4::add(vec, vec);
            v128::store(arr.as_mut_ptr() as *mut v128, vec);

            assert_eq!(arr, [0, 2, 4, 6]);
        }
    }

    test_uop!(f32x4[f32; 4] | neg[f32x4_neg_test]: [0., 1., 2., 3.] => [ 0., -1., -2., -3.]);
    test_uop!(f32x4[f32; 4] | abs[f32x4_abs_test]: [0., -1., 2., -3.] => [ 0., 1., 2., 3.]);
    test_bop!(f32x4[f32; 4] | min[f32x4_min_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., -3., -4., 8.]);
    test_bop!(f32x4[f32; 4] | min[f32x4_min_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
              => [0., -3., -4., std::f32::NAN]);
    test_bop!(f32x4[f32; 4] | max[f32x4_max_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -1., 7., 10.]);
    test_bop!(f32x4[f32; 4] | max[f32x4_max_test_nan]:
              ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
              => [1., -1., 7., std::f32::NAN]);
    test_bop!(f32x4[f32; 4] | add[f32x4_add_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -4., 3., 18.]);
    test_bop!(f32x4[f32; 4] | sub[f32x4_sub_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [-1., 2., 11., -2.]);
    test_bop!(f32x4[f32; 4] | mul[f32x4_mul_test]:
              ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., 3., -28., 80.]);
    test_bop!(f32x4[f32; 4] | div[f32x4_div_test]:
              ([0., -8., 70., 8.], [1., 4., 10., 2.]) => [0., -2., 7., 4.]);

    test_uop!(f64x2[f64; 2] | neg[f64x2_neg_test]: [0., 1.] => [ 0., -1.]);
    test_uop!(f64x2[f64; 2] | abs[f64x2_abs_test]: [0., -1.] => [ 0., 1.]);
    test_bop!(f64x2[f64; 2] | min[f64x2_min_test]:
              ([0., -1.], [1., -3.]) => [0., -3.]);
    test_bop!(f64x2[f64; 2] | min[f64x2_min_test_nan]:
              ([7., 8.], [-4., std::f64::NAN])
              => [ -4., std::f64::NAN]);
    test_bop!(f64x2[f64; 2] | max[f64x2_max_test]:
              ([0., -1.], [1., -3.]) => [1., -1.]);
    test_bop!(f64x2[f64; 2] | max[f64x2_max_test_nan]:
              ([7., 8.], [ -4., std::f64::NAN])
              => [7., std::f64::NAN]);
    test_bop!(f64x2[f64; 2] | add[f64x2_add_test]:
              ([0., -1.], [1., -3.]) => [1., -4.]);
    test_bop!(f64x2[f64; 2] | sub[f64x2_sub_test]:
              ([0., -1.], [1., -3.]) => [-1., 2.]);
    test_bop!(f64x2[f64; 2] | mul[f64x2_mul_test]:
              ([0., -1.], [1., -3.]) => [0., 3.]);
    test_bop!(f64x2[f64; 2] | div[f64x2_div_test]:
              ([0., -8.], [1., 4.]) => [0., -2.]);

    macro_rules! test_conv {
        ($test_id:ident | $conv_id:ident | $to_ty:ident | $from:expr,  $to:expr) => {
            #[wasm_bindgen_test]
            fn $test_id() {
                unsafe {
                    let from: v128 = mem::transmute($from);
                    let to: v128 = mem::transmute($to);

                    let r: v128 = $to_ty::$conv_id(from);

                    compare_bytes(r, to);
                }
            }
        };
    }

    test_conv!(
        f32x4_convert_s_i32x4 | convert_s_i32x4 | f32x4 | [1_i32, 2, 3, 4],
        [1_f32, 2., 3., 4.]
    );
    test_conv!(
        f32x4_convert_u_i32x4 | convert_u_i32x4 | f32x4 | [u32::max_value(), 2, 3, 4],
        [u32::max_value() as f32, 2., 3., 4.]
    );
    test_conv!(
        f64x2_convert_s_i64x2 | convert_s_i64x2 | f64x2 | [1_i64, 2],
        [1_f64, 2.]
    );
    test_conv!(
        f64x2_convert_u_i64x2 | convert_u_i64x2 | f64x2 | [u64::max_value(), 2],
        [18446744073709552000.0, 2.]
    );

    // FIXME: this fails, and produces -2147483648 instead of saturating at
    // i32::max_value() test_conv!(i32x4_trunc_s_f32x4_sat | trunc_s_f32x4_sat
    // | i32x4 | [1_f32, 2., (i32::max_value() as f32 + 1.), 4.],
    // [1_i32, 2, i32::max_value(), 4]); FIXME: add other saturating tests
}
