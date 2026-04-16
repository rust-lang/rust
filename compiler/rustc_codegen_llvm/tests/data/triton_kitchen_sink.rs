/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![feature(no_core)]
#![feature(intrinsics, lang_items, rustc_attrs)]
#![feature(arbitrary_self_types)]
#![feature(const_trait_impl)]
#![feature(auto_traits)]
#![no_core]
#![no_implicit_prelude]

#[lang = "freeze"]
pub unsafe auto trait Freeze {}

#[lang = "meta_sized"]
pub unsafe auto trait MetaSized {}

#[lang = "pointee_sized"]
pub unsafe auto trait PointeeSized {}

// Required language items for no_core
#[lang = "sized"]
pub trait Sized {}

#[lang = "clone"]
pub trait Clone {
    fn clone(&self) -> Self;
}

#[lang = "copy"]
pub trait Copy: Clone {}

impl<T> Copy for *const T {}
impl<T> Copy for *mut T {}
impl<T> Clone for *const T {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Clone for *mut T {
    fn clone(&self) -> Self {
        *self
    }
}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

// Required for array-to-slice coercions: &[T; N] → &[T]
#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // This function is a shim that the compiler fills in
    unsafe { drop_in_place(to_drop) }
}

// Required language items for arithmetic operations
#[lang = "panic_const_add_overflow"]
pub fn panic_const_add_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_sub_overflow"]
pub fn panic_const_sub_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_mul_overflow"]
pub fn panic_const_mul_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_div_overflow"]
pub fn panic_const_div_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_rem_overflow"]
pub fn panic_const_rem_overflow() -> ! {
    loop {}
}

#[lang = "panic_location"]
pub struct PanicLocation {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

// Also implement Copy for other primitive types that might be needed
impl Copy for i32 {}
impl Clone for i32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for f32 {}
impl Clone for f32 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for f64 {}
impl Clone for f64 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for i8 {}
impl Clone for i8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i16 {}
impl Clone for i16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i64 {}
impl Clone for i64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u8 {}
impl Clone for u8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u16 {}
impl Clone for u16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u32 {}
impl Clone for u32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u64 {}
impl Clone for u64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for usize {}
impl Clone for usize {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for bool {}
impl Clone for bool {
    fn clone(&self) -> Self {
        *self
    }
}

pub enum Option<T> {
    None,
    Some(T),
}

use Option::*;

/// Raw memory transmute — maps to the compiler `transmute` intrinsic.
#[rustc_intrinsic]
pub unsafe fn transmute<Src, Dst>(src: Src) -> Dst {
    loop {}
}

/// Create a `&[T]` from a raw pointer and a length.
/// Works in `no_core` where the array→slice coercion is unavailable.
///
/// # Safety
/// `data` must point to at least `len` consecutive initialized `T` values.
#[inline(always)]
pub unsafe fn slice_from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
    // A `&[T]` is a fat pointer `(*const T, usize)` on all current Rust targets.
    unsafe { transmute::<(*const T, usize), &[T]>((data, len)) }
}

pub const trait Into<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into(self) -> T;
}

pub const trait From<T>: Sized {
    /// Converts to this type from the input type.
    fn from(value: T) -> Self;
}

impl<T> From<T> for T {
    fn from(value: T) -> Self {
        value
    }
}

impl<T, U> Into<U> for T
where
    U: From<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

pub mod core {
    pub mod ops {
        // Arithmetic operation lang items
        #[lang = "mul"]
        pub trait Mul<RHS = Self> {
            type Output;
            fn mul(self, rhs: RHS) -> Self::Output;
        }

        impl Mul for i32 {
            type Output = i32;
            fn mul(self, rhs: i32) -> Self::Output {
                0
            }
        }

        impl Mul for i64 {
            type Output = i64;
            fn mul(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "add"]
        pub trait Add<RHS = Self> {
            type Output;
            fn add(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i32 {
            type Output = i32;
            fn add(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u32 {
            type Output = u32;
            fn add(self, rhs: u32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u64 {
            type Output = u64;
            fn add(self, rhs: u64) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i64 {
            type Output = i64;
            fn add(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "sub"]
        pub trait Sub<RHS = Self> {
            type Output;
            fn sub(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for i32 {
            type Output = i32;
            fn sub(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for u32 {
            type Output = u32;
            fn sub(self, rhs: u32) -> Self::Output {
                0
            }
        }

        #[lang = "div"]
        pub trait Div<RHS = Self> {
            type Output;
            fn div(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Div for i32 {
            type Output = i32;
            fn div(self, rhs: i32) -> Self::Output {
                0
            }
        }

        #[lang = "rem"]
        pub trait Rem<RHS = Self> {
            type Output;
            fn rem(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Rem for i32 {
            type Output = i32;
            fn rem(self, rhs: i32) -> Self::Output {
                0
            }
        }

        #[lang = "neg"]
        pub trait Neg {
            type Output;
            fn neg(self) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for i32 {
            type Output = i32;
            fn neg(self) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for f32 {
            type Output = f32;
            fn neg(self) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for f64 {
            type Output = f64;
            fn neg(self) -> Self::Output {
                0.0
            }
        }
    }
}
pub mod triton {
    pub use super::*;
    /*
     * Copyright (c) 2026 Teenygrad.
     *
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     *   http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */

    use core::ops::{Add, Div, Mul, Neg, Sub};

    use self::types::{self as ty};

    pub use types::*;

    /*------------------------------ Parameter Enums ------------------------------*/

    #[repr(i32)]
    pub enum Axis {
        X = 0,
        Y = 1,
        Z = 2,
    }

    /// Padding value applied to out-of-bounds lanes when using `boundary_check` in `load`.
    pub enum PaddingOption {
        /// Pad with zero.
        Zero,
        /// Pad with NaN.
        Nan,
    }

    /// L1/L2 cache behaviour for load and store instructions.
    pub enum CacheModifier {
        /// Cache at all levels (L1 + L2).
        Ca,
        /// Cache at global level only (L2, bypass L1).
        Cg,
        /// Volatile — don't cache, always fetch from memory.
        Cv,
        /// Write-back at all coherent levels.
        Wb,
        /// Streaming — likely accessed once.
        Cs,
    }

    /// Cache eviction priority hint for load and store instructions.
    pub enum EvictionPolicy {
        EvictFirst,
        EvictLast,
        NoEvict,
    }

    /// Tensor-core precision mode for `dot` on `f32 × f32` inputs.
    pub enum InputPrecision {
        /// TF32 precision (default on devices with Tensor Cores).
        TF32,
        /// Emulate higher precision using three TF32 dot products.
        TF32x3,
        /// Full IEEE-754 precision.
        IEEE,
    }

    /// Rounding mode used when down-casting floating-point types in `cast`.
    pub enum FpDowncastRounding {
        /// Round to nearest, ties to even.
        Rtne,
        /// Round towards zero (truncate).
        Rtz,
    }

    /// Input format for scaled dot-product (`dot_scaled`).
    pub enum DotFormat {
        E4M3,
        E5M2,
        E2M1x2,
        E2M1x4,
        BF16x2,
        Int8,
        UInt8,
    }

    /// Memory ordering semantics for atomic operations.
    pub enum MemSem {
        Relaxed,
        Acquire,
        Release,
        /// Acquire + Release (default).
        AcqRel,
    }

    /// Synchronization scope for atomic operations.
    pub enum MemScope {
        /// Cooperative thread array (thread block).
        Cta,
        /// All threads on the GPU (default).
        Gpu,
        /// All threads in the system.
        Sys,
    }

    /*------------------------------ Triton Trait ------------------------------*/

    pub trait Triton
    where
        Self::I32Tensor: Add<i32, Output = Self::I32Tensor>,
        Self::I32Tensor: Sub<i32, Output = Self::I32Tensor>,
        Self::I32Tensor: Mul<i32, Output = Self::I32Tensor>,
    {
        type BF16: ty::BF16;
        type BoolTensor: Copy + Clone;
        type I32Tensor: Copy + Clone;
        type Tensor<D: ty::Dtype>: Copy
            + Clone
            + Add<Self::Tensor<D>, Output = Self::Tensor<D>>
            + Sub<Self::Tensor<D>, Output = Self::Tensor<D>>
            + Mul<Self::Tensor<D>, Output = Self::Tensor<D>>
            + Div<Self::Tensor<D>, Output = Self::Tensor<D>>
            + Neg<Output = Self::Tensor<D>>;
        type Pointer<D: ty::Dtype>: Copy
            + Clone
            + ty::Dtype
            + Add<Self::Pointer<D>, Output = Self::Pointer<D>>;

        /*------------------------------ Programming Model ------------------------------*/

        fn program_id(axis: Axis) -> i32;

        fn num_programs(axis: Axis) -> i32;

        /*------------------------------ Creation Ops ------------------------------*/

        fn arange(start: impl Into<i32>, end: impl Into<i32>) -> Self::I32Tensor;

        fn zeros<D: ty::Dtype>(shape: &[i32]) -> Self::Tensor<D>;

        fn zeros_like<D: ty::Dtype>(x: Self::Tensor<D>) -> Self::Tensor<D>;

        fn full<D: ty::Dtype>(shape: &[i32], value: D) -> Self::Tensor<D>;

        /// Cast a tensor to a different dtype.
        ///
        /// - `fp_downcast_rounding`: rounding mode when narrowing float types (default `None` = unspecified).
        /// - `bitcast`: reinterpret bits without conversion (default `false`).
        fn cast<Src: ty::Dtype, Dst: ty::Dtype>(
            x: Self::Tensor<Src>,
            fp_downcast_rounding: Option<FpDowncastRounding>,
            bitcast: bool,
        ) -> Self::Tensor<Dst>;

        /// Concatenate two tensors.
        ///
        /// - `can_reorder`: allow the compiler to reorder elements (default `false`).
        fn cat<D: ty::Dtype>(
            a: Self::Tensor<D>,
            b: Self::Tensor<D>,
            can_reorder: bool,
        ) -> Self::Tensor<D>;

        /*------------------------------ Shape Manipulation Ops ------------------------------*/

        /// Broadcast two tensors to a common compatible shape.
        fn broadcast<D: ty::Dtype>(
            a: Self::Tensor<D>,
            b: Self::Tensor<D>,
        ) -> (Self::Tensor<D>, Self::Tensor<D>);

        fn broadcast_to<D: ty::Dtype>(x: Self::Tensor<D>, shape: &[i32]) -> Self::Tensor<D>;

        fn expand_dims<D: ty::Dtype>(x: Self::Tensor<D>, axis: i32) -> Self::Tensor<D>;

        fn permute<D: ty::Dtype>(x: Self::Tensor<D>, dims: &[i32]) -> Self::Tensor<D>;

        /// Reshape a tensor.
        ///
        /// - `can_reorder`: allow element reordering during reshape (default `false`).
        fn reshape<D: ty::Dtype>(
            x: Self::Tensor<D>,
            shape: &[i32],
            can_reorder: bool,
        ) -> Self::Tensor<D>;

        /// Permute dimensions. Alias for `permute`.
        fn trans<D: ty::Dtype>(x: Self::Tensor<D>, dims: &[i32]) -> Self::Tensor<D>;

        /// Flatten to 1-D.
        ///
        /// - `can_reorder`: allow element reordering (default `false`).
        fn ravel<D: ty::Dtype>(x: Self::Tensor<D>, can_reorder: bool) -> Self::Tensor<D>;

        /// View with a new shape (order not preserved).
        fn view<D: ty::Dtype>(x: Self::Tensor<D>, shape: &[i32]) -> Self::Tensor<D>;

        /// Join two tensors along a new minor dimension.
        fn join<D: ty::Dtype>(a: Self::Tensor<D>, b: Self::Tensor<D>) -> Self::Tensor<D>;

        /// Interleave two tensors along their last dimension.
        fn interleave<D: ty::Dtype>(a: Self::Tensor<D>, b: Self::Tensor<D>) -> Self::Tensor<D>;

        /// Split a tensor in two along its last dimension (which must have size 2).
        fn split<D: ty::Dtype>(x: Self::Tensor<D>) -> (Self::Tensor<D>, Self::Tensor<D>);

        /*------------------------------ Linear Algebra Ops ------------------------------*/

        /// Matrix (or batched matrix) multiply.
        ///
        /// - `acc`: optional accumulator tensor added to the result.
        /// - `input_precision`: Tensor Core precision for `f32 × f32` (default `None` = TF32 on capable hardware).
        /// - `max_num_imprecise_acc`: limit on imprecise accumulations (default `None`).
        fn dot<D: ty::Num, O: ty::Num>(
            a: Self::Tensor<D>,
            b: Self::Tensor<D>,
            acc: Option<Self::Tensor<O>>,
            input_precision: Option<InputPrecision>,
            max_num_imprecise_acc: Option<i32>,
        ) -> Self::Tensor<O>;

        /// Scaled mixed-precision matrix multiply (FP8 / narrow formats).
        ///
        /// - `acc`: optional accumulator (default `None`).
        /// - `fast_math`: allow reduced precision accumulation (default `false`).
        fn dot_scaled<D: ty::Num, S: ty::Num, O: ty::Num>(
            lhs: Self::Tensor<D>,
            lhs_scale: Self::Tensor<S>,
            lhs_format: DotFormat,
            rhs: Self::Tensor<D>,
            rhs_scale: Self::Tensor<S>,
            rhs_format: DotFormat,
            acc: Option<Self::Tensor<O>>,
            fast_math: bool,
        ) -> Self::Tensor<O>;

        /*------------------------------ Memory / Pointer Ops ------------------------------*/

        /// Create a block pointer encoding shape, strides, offsets, and tile shape.
        fn make_block_ptr<D: ty::Dtype>(
            base: Self::Pointer<D>,
            shape: &[i32],
            strides: &[i32],
            offsets: &[i32],
            block_shape: &[i32],
            order: &[i32],
        ) -> Self::Pointer<D>;

        /// Advance a block pointer by the given per-dimension offsets.
        fn advance<D: ty::Dtype>(ptr: Self::Pointer<D>, offsets: &[i32]) -> Self::Pointer<D>;

        /// Create a tensor descriptor for TMA (Tensor Memory Accelerator) operations.
        ///
        /// - `padding_option`: out-of-bounds padding behaviour (default `PaddingOption::Zero`).
        fn make_tensor_descriptor<D: ty::Dtype>(
            base: Self::Pointer<D>,
            shape: &[i32],
            strides: &[i32],
            block_shape: &[i32],
            padding_option: Option<PaddingOption>,
        ) -> Self::Pointer<D>;

        /// Load a tile from memory using a tensor descriptor and per-dimension offsets.
        fn load_tensor_descriptor<D: ty::Dtype>(
            desc: Self::Pointer<D>,
            offsets: &[i32],
        ) -> Self::Tensor<D>;

        /// Store a tile to memory using a tensor descriptor and per-dimension offsets.
        fn store_tensor_descriptor<D: ty::Dtype>(
            desc: Self::Pointer<D>,
            offsets: &[i32],
            value: Self::Tensor<D>,
        );

        /// Load a tensor from memory.
        ///
        /// - `mask`: when `Some`, lanes where mask is `false` are not loaded (default `None` = unconditional).
        /// - `other`: fill value for masked-off lanes (default `None` = undefined).
        /// - `boundary_check`: dimensions to check for out-of-bounds (block-pointer mode only, default `&[]`).
        /// - `padding_option`: fill for out-of-bounds lanes in block-pointer mode (default `None`).
        /// - `cache_modifier`: L1/L2 cache behaviour (default `None`).
        /// - `eviction_policy`: eviction priority hint (default `None`).
        /// - `volatile`: always fetch fresh from memory (default `false`).
        fn load<D: ty::Dtype, const N: usize>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            mask: Option<Self::BoolTensor>,
            other: Option<Self::Tensor<D>>,
            boundary_check: &[i32; N],
            padding_option: Option<PaddingOption>,
            cache_modifier: Option<CacheModifier>,
            eviction_policy: Option<EvictionPolicy>,
            volatile: bool,
        ) -> Self::Tensor<D>;

        /// Store a tensor to memory.
        ///
        /// - `mask`: when `Some`, lanes where mask is `false` are not stored (default `None` = unconditional).
        /// - `boundary_check`: dimensions to check for out-of-bounds (block-pointer mode only, default `&[]`).
        /// - `cache_modifier`: L1/L2 cache behaviour (default `None`).
        /// - `eviction_policy`: eviction priority hint (default `None`).
        fn store<D: ty::Dtype, const N: usize>(
            dest: Self::Tensor<Self::Pointer<D>>,
            src: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            boundary_check: &[i32; N],
            cache_modifier: Option<CacheModifier>,
            eviction_policy: Option<EvictionPolicy>,
        );

        /*------------------------------ Indexing Ops ------------------------------*/

        /// Conditional element selection — corresponds to `tl.where`.
        /// Named `where_` to avoid collision with the Rust keyword `where`.
        fn where_<D: ty::Dtype>(
            cond: Self::BoolTensor,
            x: Self::Tensor<D>,
            y: Self::Tensor<D>,
        ) -> Self::Tensor<D>;

        /// Reverse a tensor along `dim`. `None` reverses all dimensions.
        fn flip<D: ty::Dtype>(x: Self::Tensor<D>, dim: Option<i32>) -> Self::Tensor<D>;

        fn gather<D: ty::Dtype>(
            src: Self::Tensor<D>,
            index: Self::I32Tensor,
            axis: i32,
        ) -> Self::Tensor<D>;

        /*------------------------------ Math Ops — Unary ------------------------------*/

        /// Element-wise absolute value.
        fn abs<D: ty::Dtype>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn ceil<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn floor<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn cos<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn sin<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn exp<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn exp2<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn log<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn log2<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn rsqrt<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn sigmoid<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn sqrt<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn sqrt_rn<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
        fn erf<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;

        /*------------------------------ Math Ops — Float (higher-level) ------------------------------*/

        /// Numerically-stable softmax along `dim`. `dim = None` defaults to the last dimension.
        ///
        /// - `keep_dims`: retain the reduced dimension with length 1 (default `false`).
        /// - `ieee_rounding`: use IEEE-754 rounding (default `false`).
        fn softmax<D: ty::Float>(
            x: Self::Tensor<D>,
            dim: Option<i32>,
            keep_dims: bool,
            ieee_rounding: bool,
        ) -> Self::Tensor<D>;

        /*------------------------------ Math Ops — Binary ------------------------------*/

        fn maximum<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;
        fn minimum<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;

        fn clamp<D: ty::Num>(
            x: Self::Tensor<D>,
            lo: Self::Tensor<D>,
            hi: Self::Tensor<D>,
        ) -> Self::Tensor<D>;

        fn fma<D: ty::Float>(
            x: Self::Tensor<D>,
            y: Self::Tensor<D>,
            z: Self::Tensor<D>,
        ) -> Self::Tensor<D>;

        fn fdiv<D: ty::Float>(
            x: Self::Tensor<D>,
            y: Self::Tensor<D>,
            ieee_rounding: bool,
        ) -> Self::Tensor<D>;

        fn div_rn<D: ty::Float>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;

        fn umulhi(x: Self::Tensor<u32>, y: Self::Tensor<u32>) -> Self::Tensor<u32>;

        /// Ceiling integer division: `ceil(x / div)`.
        fn cdiv(x: i32, div: i32) -> i32;

        /// Swizzle 2-D indices for shared-memory bank-conflict avoidance.
        /// Returns the remapped `(i, j)` indices.
        fn swizzle2d(i: i32, j: i32, size_i: i32, size_j: i32, size_g: i32) -> (i32, i32);

        /*------------------------------ Reduction Ops ------------------------------*/

        /// Sum all elements along `axis`. `axis = None` reduces all dimensions.
        fn sum<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: Option<i32>,
            keep_dims: bool,
        ) -> Self::Tensor<D>;

        /// Maximum along `axis`. `axis = None` reduces all dimensions.
        fn max<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: Option<i32>,
            keep_dims: bool,
        ) -> Self::Tensor<D>;

        /// Maximum along `axis`, also returning the index of the maximum.
        ///
        /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
        fn max_with_indices<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: i32,
            tie_break_left: bool,
            keep_dims: bool,
        ) -> (Self::Tensor<D>, Self::I32Tensor);

        /// Minimum along `axis`. `axis = None` reduces all dimensions.
        fn min<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: Option<i32>,
            keep_dims: bool,
        ) -> Self::Tensor<D>;

        /// Minimum along `axis`, also returning the index of the minimum.
        ///
        /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
        fn min_with_indices<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: i32,
            tie_break_left: bool,
            keep_dims: bool,
        ) -> (Self::Tensor<D>, Self::I32Tensor);

        /// Index of the maximum along `axis`.
        ///
        /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
        fn argmax<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: i32,
            tie_break_left: bool,
            keep_dims: bool,
        ) -> Self::I32Tensor;

        /// Index of the minimum along `axis`.
        ///
        /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
        fn argmin<D: ty::Num>(
            x: Self::Tensor<D>,
            axis: i32,
            tie_break_left: bool,
            keep_dims: bool,
        ) -> Self::I32Tensor;

        /// XOR-reduction along `axis`. `axis = None` reduces all dimensions.
        fn xor_sum<D: ty::Int>(
            x: Self::Tensor<D>,
            axis: Option<i32>,
            keep_dims: bool,
        ) -> Self::Tensor<D>;

        /*------------------------------ Scan / Sort Ops ------------------------------*/

        fn cumsum<D: ty::Num>(x: Self::Tensor<D>, axis: i32, reverse: bool) -> Self::Tensor<D>;

        fn cumprod<D: ty::Num>(x: Self::Tensor<D>, axis: i32, reverse: bool) -> Self::Tensor<D>;

        /// Sort along `dim`. `dim = None` sorts along the last dimension.
        fn sort<D: ty::Num>(
            x: Self::Tensor<D>,
            dim: Option<i32>,
            descending: bool,
        ) -> Self::Tensor<D>;

        /// Compute a histogram with `num_bins` bins (width 1, starting at 0).
        ///
        /// - `mask`: when `Some`, masked-off elements are excluded (default `None`).
        fn histogram(
            x: Self::I32Tensor,
            num_bins: i32,
            mask: Option<Self::BoolTensor>,
        ) -> Self::I32Tensor;

        /// Generic reduction along `axis` using a user-supplied combine function.
        ///
        /// `combine_fn` must be a statically-known function pointer (corresponds to a
        /// `@triton.jit`-decorated helper in Python Triton).
        fn reduce<D: ty::Dtype, O: ty::Dtype>(
            x: Self::Tensor<D>,
            axis: i32,
            combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>,
            keep_dims: bool,
        ) -> Self::Tensor<O>;

        /// Generic prefix-scan along `axis` using a user-supplied combine function.
        ///
        /// - `reverse`: scan in the reverse direction (default `false`).
        fn associative_scan<D: ty::Dtype>(
            x: Self::Tensor<D>,
            axis: i32,
            combine_fn: fn(Self::Tensor<D>, Self::Tensor<D>) -> Self::Tensor<D>,
            reverse: bool,
        ) -> Self::Tensor<D>;

        /*------------------------------ Atomic Ops ------------------------------*/

        /// Atomic add. Returns the previous value.
        ///
        /// - `mask`: when `Some`, only masked lanes perform the operation (default `None`).
        /// - `sem`: memory ordering semantics (default `None` = AcqRel).
        /// - `scope`: synchronization scope (default `None` = Gpu).
        fn atomic_add<D: ty::Num>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_and<D: ty::Int>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_or<D: ty::Int>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_xor<D: ty::Int>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_max<D: ty::Num>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_min<D: ty::Num>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        fn atomic_xchg<D: ty::Dtype>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            val: Self::Tensor<D>,
            mask: Option<Self::BoolTensor>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        /// Atomic compare-and-swap. Returns the previous value.
        fn atomic_cas<D: ty::Dtype>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            cmp: Self::Tensor<D>,
            val: Self::Tensor<D>,
            sem: Option<MemSem>,
            scope: Option<MemScope>,
        ) -> Self::Tensor<D>;

        /*------------------------------ Random Number Generation ------------------------------*/

        /// Uniform random `f32` in `[0, 1)`.
        ///
        /// - `n_rounds`: number of Philox rounds (default `10`).
        fn rand(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::Tensor<f32>;

        /// Standard-normal random `f32`.
        ///
        /// - `n_rounds`: number of Philox rounds (default `10`).
        fn randn(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::Tensor<f32>;

        /// Random `i32`.
        ///
        /// - `n_rounds`: number of Philox rounds (default `10`).
        fn randint(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::I32Tensor;

        /// Four random `i32` streams (maximally efficient Philox entry point).
        ///
        /// - `n_rounds`: number of Philox rounds (default `10`).
        fn randint4x(
            seed: u32,
            offsets: Self::I32Tensor,
            n_rounds: i32,
        ) -> (Self::I32Tensor, Self::I32Tensor, Self::I32Tensor, Self::I32Tensor);

        /*------------------------------ Inline Assembly ------------------------------*/

        /// Emit inline PTX/assembly applied element-wise across a tensor.
        ///
        /// - `asm`: the assembly template string.
        /// - `constraints`: register constraint string.
        /// - `is_pure`: whether the assembly has no side-effects (may be CSE'd).
        /// - `pack`: number of elements packed into each register.
        fn inline_asm_elementwise<D: ty::Dtype>(
            asm: &str,
            constraints: &str,
            is_pure: bool,
            pack: i32,
        ) -> Self::Tensor<D>;

        /*------------------------------ Compiler Hint Ops ------------------------------*/

        /// Assert that `cond` is always true, allowing the compiler to assume so.
        fn assume(cond: Self::BoolTensor);

        /// Hint that values of `x` are always multiples of the given constants.
        fn multiple_of<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

        /// Hint that `x` has `values[i]` contiguous elements along dimension `i`.
        fn max_contiguous<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

        /// Hint that `x` has `values[i]` constant elements along dimension `i`.
        fn max_constancy<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

        /*------------------------------ Debug Ops ------------------------------*/

        /// Insert a memory barrier for debugging purposes.
        fn debug_barrier();

        /// Emit a runtime assertion on the device. No-op when `cond` is `true`.
        ///
        /// - `msg`: message shown on assertion failure (default `""`).
        /// - `mask`: when `Some`, only lanes where mask is `true` check the assertion.
        fn device_assert(cond: Self::BoolTensor, msg: &str, mask: Option<Self::BoolTensor>);

        /// Print a tensor value from device code for debugging.
        ///
        /// - `hex`: print values in hexadecimal (default `false`).
        fn device_print<D: ty::Dtype>(prefix: &str, val: Self::Tensor<D>, hex: bool);

        /// Compile-time assertion (evaluated before kernel launch).
        fn static_assert(cond: bool, msg: &str);

        /// Compile-time print (evaluated before kernel launch).
        fn static_print(msg: &str);
    }
    pub mod llvm {
        pub use super::super::*;
        /*
         * Copyright (c) 2026 Teenygrad.
         *
         * Licensed under the Apache License, Version 2.0 (the "License");
         * you may not use this file except in compliance with the License.
         * You may obtain a copy of the License at
         *
         *   http://www.apache.org/licenses/LICENSE-2.0
         *
         * Unless required by applicable law or agreed to in writing, software
         * distributed under the License is distributed on an "AS IS" BASIS,
         * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         * See the License for the specific language governing permissions and
         * limitations under the License.
         */

        pub mod triton {
            pub use super::super::super::*;
            /*
             * Copyright (c) 2026 Teenygrad.
             *
             * Licensed under the Apache License, Version 2.0 (the "License");
             * you may not use this file except in compliance with the License.
             * You may obtain a copy of the License at
             *
             *   http://www.apache.org/licenses/LICENSE-2.0
             *
             * Unless required by applicable law or agreed to in writing, software
             * distributed under the License is distributed on an "AS IS" BASIS,
             * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
             * See the License for the specific language governing permissions and
             * limitations under the License.
             */

            use super::super::Triton;
            use super::super::{
                Axis, CacheModifier, DotFormat, EvictionPolicy, FpDowncastRounding, InputPrecision,
                MemScope, MemSem, PaddingOption, types as ty,
            };

            pub struct LlvmTriton {}

            impl Triton for LlvmTriton {
                type BF16 = num::BF16;
                type BoolTensor = tensor::BoolTensor;
                type I32Tensor = tensor::I32Tensor;
                type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
                type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

                /*------------------------------ Programming Model ------------------------------*/

                #[inline(never)]
                fn program_id(_axis: Axis) -> i32 {
                    0
                }

                #[inline(never)]
                fn num_programs(_axis: Axis) -> i32 {
                    0
                }

                /*------------------------------ Creation Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn arange(_start: impl Into<i32>, _end: impl Into<i32>) -> Self::I32Tensor {
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn zeros<D: ty::Dtype>(_shape: &[i32]) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn zeros_like<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn full<D: ty::Dtype>(_shape: &[i32], _value: D) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cast<Src: ty::Dtype, Dst: ty::Dtype>(
                    _x: Self::Tensor<Src>,
                    _fp_downcast_rounding: Option<FpDowncastRounding>,
                    _bitcast: bool,
                ) -> Self::Tensor<Dst> {
                    tensor::Tensor(0 as *mut Dst)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cat<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                    _can_reorder: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Shape Manipulation Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn broadcast<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                ) -> (Self::Tensor<D>, Self::Tensor<D>) {
                    let t = tensor::Tensor(0 as *mut D);
                    (t, t)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn broadcast_to<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _shape: &[i32],
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn expand_dims<D: ty::Dtype>(_x: Self::Tensor<D>, _axis: i32) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn permute<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn reshape<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _shape: &[i32],
                    _can_reorder: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn trans<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ravel<D: ty::Dtype>(_x: Self::Tensor<D>, _can_reorder: bool) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn view<D: ty::Dtype>(_x: Self::Tensor<D>, _shape: &[i32]) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn join<D: ty::Dtype>(_a: Self::Tensor<D>, _b: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn interleave<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn split<D: ty::Dtype>(_x: Self::Tensor<D>) -> (Self::Tensor<D>, Self::Tensor<D>) {
                    let t = tensor::Tensor(0 as *mut D);
                    (t, t)
                }

                /*------------------------------ Linear Algebra Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn dot_scaled<D: ty::Num, S: ty::Num, O: ty::Num>(
                    _lhs: Self::Tensor<D>,
                    _lhs_scale: Self::Tensor<S>,
                    _lhs_format: DotFormat,
                    _rhs: Self::Tensor<D>,
                    _rhs_scale: Self::Tensor<S>,
                    _rhs_format: DotFormat,
                    _acc: Option<Self::Tensor<O>>,
                    _fast_math: bool,
                ) -> Self::Tensor<O> {
                    tensor::Tensor(0 as *mut O)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn dot<D: ty::Num, O: ty::Num>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                    _acc: Option<Self::Tensor<O>>,
                    _input_precision: Option<InputPrecision>,
                    _max_num_imprecise_acc: Option<i32>,
                ) -> Self::Tensor<O> {
                    tensor::Tensor(0 as *mut O)
                }

                /*------------------------------ Memory / Pointer Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn make_block_ptr<D: ty::Dtype>(
                    _base: Self::Pointer<D>,
                    _shape: &[i32],
                    _strides: &[i32],
                    _offsets: &[i32],
                    _block_shape: &[i32],
                    _order: &[i32],
                ) -> Self::Pointer<D> {
                    pointer::Pointer(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn advance<D: ty::Dtype>(
                    _ptr: Self::Pointer<D>,
                    _offsets: &[i32],
                ) -> Self::Pointer<D> {
                    pointer::Pointer(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn make_tensor_descriptor<D: ty::Dtype>(
                    _base: Self::Pointer<D>,
                    _shape: &[i32],
                    _strides: &[i32],
                    _block_shape: &[i32],
                    _padding_option: Option<PaddingOption>,
                ) -> Self::Pointer<D> {
                    pointer::Pointer(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn load_tensor_descriptor<D: ty::Dtype>(
                    _desc: Self::Pointer<D>,
                    _offsets: &[i32],
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                fn store_tensor_descriptor<D: ty::Dtype>(
                    _desc: Self::Pointer<D>,
                    _offsets: &[i32],
                    _value: Self::Tensor<D>,
                ) {
                    // nop
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn load<D: ty::Dtype, const N: usize>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _mask: Option<Self::BoolTensor>,
                    _other: Option<Self::Tensor<D>>,
                    _boundary_check: &[i32; N],
                    _padding_option: Option<PaddingOption>,
                    _cache_modifier: Option<CacheModifier>,
                    _eviction_policy: Option<EvictionPolicy>,
                    _volatile: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                fn store<D: ty::Dtype, const N: usize>(
                    _dest: Self::Tensor<Self::Pointer<D>>,
                    _src: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _boundary_check: &[i32; N],
                    _cache_modifier: Option<CacheModifier>,
                    _eviction_policy: Option<EvictionPolicy>,
                ) {
                    // nop
                }

                /*------------------------------ Indexing Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn where_<D: ty::Dtype>(
                    _cond: Self::BoolTensor,
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn flip<D: ty::Dtype>(_x: Self::Tensor<D>, _dim: Option<i32>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn gather<D: ty::Dtype>(
                    _src: Self::Tensor<D>,
                    _index: Self::I32Tensor,
                    _axis: i32,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Math Ops — Unary (floating-point) ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn abs<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ceil<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn floor<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cos<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sin<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn exp<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn exp2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn log<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn log2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn rsqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sigmoid<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sqrt_rn<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn erf<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn softmax<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _dim: Option<i32>,
                    _keep_dims: bool,
                    _ieee_rounding: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Math Ops — Binary ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn maximum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn minimum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn clamp<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _lo: Self::Tensor<D>,
                    _hi: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn fma<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                    _z: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn fdiv<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                    _ieee_rounding: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn div_rn<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn umulhi(_x: Self::Tensor<u32>, _y: Self::Tensor<u32>) -> Self::Tensor<u32> {
                    tensor::Tensor(0 as *mut u32)
                }

                #[inline(never)]
                fn cdiv(_x: i32, _div: i32) -> i32 {
                    0
                }

                #[inline(never)]
                fn swizzle2d(
                    _i: i32,
                    _j: i32,
                    _size_i: i32,
                    _size_j: i32,
                    _size_g: i32,
                ) -> (i32, i32) {
                    (0, 0)
                }

                /*------------------------------ Reduction Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn max<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn max_with_indices<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> (Self::Tensor<D>, Self::I32Tensor) {
                    (tensor::Tensor(0 as *mut D), tensor::Tensor(0 as *mut i32))
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn min<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn min_with_indices<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> (Self::Tensor<D>, Self::I32Tensor) {
                    (tensor::Tensor(0 as *mut D), tensor::Tensor(0 as *mut i32))
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn argmax<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> Self::I32Tensor {
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn argmin<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> Self::I32Tensor {
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn xor_sum<D: ty::Int>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Scan / Sort Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cumsum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cumprod<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sort<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _dim: Option<i32>,
                    _descending: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn histogram(
                    _x: Self::I32Tensor,
                    _num_bins: i32,
                    _mask: Option<Self::BoolTensor>,
                ) -> Self::I32Tensor {
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn reduce<D: ty::Dtype, O: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>,
                    _keep_dims: bool,
                ) -> Self::Tensor<O> {
                    tensor::Tensor(0 as *mut O)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn associative_scan<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _combine_fn: fn(Self::Tensor<D>, Self::Tensor<D>) -> Self::Tensor<D>,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Atomic Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_add<D: ty::Num>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_and<D: ty::Int>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_or<D: ty::Int>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_xor<D: ty::Int>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_max<D: ty::Num>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_min<D: ty::Num>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_xchg<D: ty::Dtype>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _val: Self::Tensor<D>,
                    _mask: Option<Self::BoolTensor>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn atomic_cas<D: ty::Dtype>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _cmp: Self::Tensor<D>,
                    _val: Self::Tensor<D>,
                    _sem: Option<MemSem>,
                    _scope: Option<MemScope>,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Random Number Generation ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn rand(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::Tensor<f32> {
                    tensor::Tensor(0 as *mut f32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randn(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::Tensor<f32> {
                    tensor::Tensor(0 as *mut f32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randint(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::I32Tensor {
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randint4x(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> (Self::I32Tensor, Self::I32Tensor, Self::I32Tensor, Self::I32Tensor)
                {
                    let t = tensor::Tensor(0 as *mut i32);
                    (t, t, t, t)
                }

                /*------------------------------ Compiler Hint Ops ------------------------------*/

                #[inline(always)]
                fn multiple_of<D: ty::Dtype>(
                    x: Self::Tensor<D>,
                    _values: &[i32],
                ) -> Self::Tensor<D> {
                    x
                }

                #[inline(always)]
                fn max_contiguous<D: ty::Dtype>(
                    x: Self::Tensor<D>,
                    _values: &[i32],
                ) -> Self::Tensor<D> {
                    x
                }

                #[inline(always)]
                fn max_constancy<D: ty::Dtype>(
                    x: Self::Tensor<D>,
                    _values: &[i32],
                ) -> Self::Tensor<D> {
                    x
                }

                /*------------------------------ Inline Assembly ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn inline_asm_elementwise<D: ty::Dtype>(
                    _asm: &str,
                    _constraints: &str,
                    _is_pure: bool,
                    _pack: i32,
                ) -> Self::Tensor<D> {
                    tensor::Tensor(0 as *mut D)
                }

                /*------------------------------ Compiler Hint Ops ------------------------------*/

                #[inline(always)]
                fn assume(_cond: Self::BoolTensor) {
                    // hint only — no-op in dummy implementation
                }

                /*------------------------------ Debug Ops ------------------------------*/

                #[inline(always)]
                fn debug_barrier() {
                    // no-op in dummy implementation
                }

                #[inline(always)]
                fn device_assert(
                    _cond: Self::BoolTensor,
                    _msg: &str,
                    _mask: Option<Self::BoolTensor>,
                ) {
                    // no-op in dummy implementation
                }

                #[inline(always)]
                fn device_print<D: ty::Dtype>(_prefix: &str, _val: Self::Tensor<D>, _hex: bool) {
                    // no-op in dummy implementation
                }

                #[inline(always)]
                fn static_assert(_cond: bool, _msg: &str) {
                    // no-op in dummy implementation (compiler lowering handles this)
                }

                #[inline(always)]
                fn static_print(_msg: &str) {
                    // no-op in dummy implementation (compiler lowering handles this)
                }
            }
            pub mod tensor {
                pub use super::super::super::*;
                /*
                 * Copyright (c) 2026 Teenygrad.
                 *
                 * Licensed under the Apache License, Version 2.0 (the "License");
                 * you may not use this file except in compliance with the License.
                 * You may obtain a copy of the License at
                 *
                 *   http://www.apache.org/licenses/LICENSE-2.0
                 *
                 * Unless required by applicable law or agreed to in writing, software
                 * distributed under the License is distributed on an "AS IS" BASIS,
                 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 * See the License for the specific language governing permissions and
                 * limitations under the License.
                 */

                use core::ops::{Add, Div, Mul, Neg, Sub};

                use super::super::super::types::{self as ty};

                /*--------------------------------- Tensor ---------------------------------*/

                pub struct Tensor<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for Tensor<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for Tensor<D> {}

                impl<D: ty::Dtype, const RANK: usize> ty::RankedTensor<D, RANK> for Tensor<D> {
                    const SHAPE: [usize; RANK] = [0; RANK];
                }
                impl<D: ty::Dtype, const RANK: usize> ty::Tensor<D, RANK> for Tensor<D> {}

                impl<D: ty::Dtype> Add<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: Tensor<D>) -> Self::Output {
                        Tensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Sub<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn sub(self, _rhs: Tensor<D>) -> Self::Output {
                        Tensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Mul<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn mul(self, _rhs: Tensor<D>) -> Self::Output {
                        Tensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Div<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn div(self, _rhs: Tensor<D>) -> Self::Output {
                        Tensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Neg for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn neg(self) -> Self::Output {
                        Tensor(0 as *mut D)
                    }
                }

                pub type BoolTensor = Tensor<bool>;
                impl<const RANK: usize> ty::BoolTensor<RANK> for BoolTensor {}

                pub type I32Tensor = Tensor<i32>;

                impl<const RANK: usize> ty::I32Tensor<RANK> for I32Tensor {}

                impl<D: ty::Num, const RANK: usize> ty::Comparison<D, RANK> for Tensor<D> {
                    type BoolTensor = BoolTensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn lt(self, _other: D) -> Self::BoolTensor {
                        Tensor(0 as *mut bool)
                    }
                }

                impl Add<i32> for I32Tensor {
                    type Output = I32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: i32) -> Self::Output {
                        Tensor(0 as *mut i32)
                    }
                }

                impl Sub<i32> for I32Tensor {
                    type Output = I32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn sub(self, _rhs: i32) -> Self::Output {
                        Tensor(0 as *mut i32)
                    }
                }

                impl Mul<i32> for I32Tensor {
                    type Output = I32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn mul(self, _rhs: i32) -> Self::Output {
                        Tensor(0 as *mut i32)
                    }
                }
            }
            pub mod types {
                pub use super::super::super::*;
                /*
                 * Copyright (c) 2026 Teenygrad.
                 *
                 * Licensed under the Apache License, Version 2.0 (the "License");
                 * you may not use this file except in compliance with the License.
                 * You may obtain a copy of the License at
                 *
                 *   http://www.apache.org/licenses/LICENSE-2.0
                 *
                 * Unless required by applicable law or agreed to in writing, software
                 * distributed under the License is distributed on an "AS IS" BASIS,
                 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 * See the License for the specific language governing permissions and
                 * limitations under the License.
                 */

                use core::ops::Add;

                // Dtype — base marker trait for all types that can flow through the system
                pub trait Dtype: Copy + Clone {}

                // Num — numeric scalars; BITS is used for device buffer allocation
                pub trait Num: Dtype {
                    const BITS: u8;
                }

                pub trait Float: Num {}
                pub trait Int: Num {}
                pub trait Bool: Dtype + Copy {}

                // Floating-point specialisations
                pub trait F8E4M3FN: Float {}
                pub trait F8E4M3FNUZ: Float {}
                pub trait F8E5M2: Float {}
                pub trait F8E5M2FNUZ: Float {}
                pub trait BF16: Float {}

                // Integer specialisations
                pub trait I4: Int {}

                // Primitive impls
                impl Dtype for bool {}

                impl Dtype for i8 {}
                impl Num for i8 {
                    const BITS: u8 = 8;
                }
                impl Int for i8 {}

                impl Dtype for i16 {}
                impl Num for i16 {
                    const BITS: u8 = 16;
                }
                impl Int for i16 {}

                impl Dtype for i32 {}
                impl Num for i32 {
                    const BITS: u8 = 32;
                }
                impl Int for i32 {}

                impl Dtype for i64 {}
                impl Num for i64 {
                    const BITS: u8 = 64;
                }
                impl Int for i64 {}

                impl Dtype for u8 {}
                impl Num for u8 {
                    const BITS: u8 = 8;
                }
                impl Int for u8 {}

                impl Dtype for u16 {}
                impl Num for u16 {
                    const BITS: u8 = 16;
                }
                impl Int for u16 {}

                impl Dtype for u32 {}
                impl Num for u32 {
                    const BITS: u8 = 32;
                }
                impl Int for u32 {}

                impl Dtype for u64 {}
                impl Num for u64 {
                    const BITS: u8 = 64;
                }
                impl Int for u64 {}

                impl Dtype for f32 {}
                impl Num for f32 {
                    const BITS: u8 = 32;
                }
                impl Float for f32 {}

                impl Dtype for f64 {}
                impl Num for f64 {
                    const BITS: u8 = 64;
                }
                impl Float for f64 {}

                // Tensor
                pub trait RankedTensor<D: Dtype, const RANK: usize>: Clone {
                    const SHAPE: [usize; RANK];
                }

                pub trait Tensor<D: Dtype, const RANK: usize>: RankedTensor<D, RANK> {}

                /// Marker trait for eager (non-symbolic) tensors.
                ///
                /// Implement this on any tensor type that computes eagerly. The generic
                /// `Layer<T>` impls (Relu, Linear, Softmax, …) are gated on this marker so
                /// that the specific `Layer<SymTensor>` impls in `nn::graph` don't conflict.
                pub trait EagerTensor {}

                pub trait BoolTensor<const RANK: usize>: Tensor<bool, RANK> {}

                pub trait Comparison<I: Num, const RANK: usize> {
                    type BoolTensor: BoolTensor<RANK>;

                    fn lt(self, other: I) -> Self::BoolTensor;
                }

                pub trait I32Tensor<const RANK: usize>:
                    Tensor<i32, RANK> + Add<i32> + Comparison<i32, RANK>
                {
                }

                // Offsets trait for adding tensor offsets to pointers
                pub trait AddOffsets<I: Int, const RANK: usize, T: Tensor<I, RANK>> {
                    type Output;

                    fn add_offsets(self, offsets: T) -> Self::Output;
                }

                // Pointer — Dtype itself (can be stored in tensors), no BITS needed
                pub trait Pointer<D: Dtype, const RANK: usize>:
                    Sized
                    + Copy
                    + Clone
                    + Dtype
                    + AddOffsets<i32, RANK, Self::I32Tensor>
                    + Add<Self>
                {
                    type I32Tensor: I32Tensor<RANK>;
                }
            }
            pub mod num {
                pub use super::super::super::*;
                /*
                 * Copyright (c) 2026 Teenygrad.
                 *
                 * Licensed under the Apache License, Version 2.0 (the "License");
                 * you may not use this file except in compliance with the License.
                 * You may obtain a copy of the License at
                 *
                 *   http://www.apache.org/licenses/LICENSE-2.0
                 *
                 * Unless required by applicable law or agreed to in writing, software
                 * distributed under the License is distributed on an "AS IS" BASIS,
                 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 * See the License for the specific language governing permissions and
                 * limitations under the License.
                 */

                use super::super::super::types as ty;

                /*--------------------------------- BF16 ---------------------------------*/

                pub struct BF16;
                impl Copy for BF16 {}
                impl Clone for BF16 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for BF16 {}
                impl ty::Num for BF16 {
                    const BITS: u8 = 16;
                }
                impl ty::Float for BF16 {}
                impl ty::BF16 for BF16 {}
            }
            pub mod pointer {
                pub use super::super::super::*;
                /*
                 * Copyright (c) 2026 Teenygrad.
                 *
                 * Licensed under the Apache License, Version 2.0 (the "License");
                 * you may not use this file except in compliance with the License.
                 * You may obtain a copy of the License at
                 *
                 *   http://www.apache.org/licenses/LICENSE-2.0
                 *
                 * Unless required by applicable law or agreed to in writing, software
                 * distributed under the License is distributed on an "AS IS" BASIS,
                 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                 * See the License for the specific language governing permissions and
                 * limitations under the License.
                 */

                use core::ops::Add;

                use crate::triton::llvm::triton::tensor::{I32Tensor, Tensor};

                use super::super::super::types::{self as ty};

                pub struct Pointer<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for Pointer<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for Pointer<D> {}

                impl<D: ty::Dtype> ty::Dtype for Pointer<D> {}

                impl<D: ty::Dtype, const RANK: usize> ty::Pointer<D, RANK> for Pointer<D> {
                    type I32Tensor = I32Tensor;
                }

                // Implement AddOffsets for Pointer
                impl<D: ty::Dtype, const RANK: usize> ty::AddOffsets<i32, RANK, I32Tensor> for Pointer<D> {
                    type Output = Tensor<Self>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add_offsets(self, _offsets: I32Tensor) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut Self)
                    }
                }

                impl<D: ty::Dtype> Add<Pointer<D>> for Pointer<D> {
                    type Output = Self;

                    #[inline(never)]
                    fn add(self, _other: Pointer<D>) -> Self::Output {
                        // dummy implementation not used in final output
                        self
                    }
                }
            }
        }
    }
    pub mod types {
        pub use super::*;
        /*
         * Copyright (c) 2026 Teenygrad.
         *
         * Licensed under the Apache License, Version 2.0 (the "License");
         * you may not use this file except in compliance with the License.
         * You may obtain a copy of the License at
         *
         *   http://www.apache.org/licenses/LICENSE-2.0
         *
         * Unless required by applicable law or agreed to in writing, software
         * distributed under the License is distributed on an "AS IS" BASIS,
         * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         * See the License for the specific language governing permissions and
         * limitations under the License.
         */

        use core::ops::Add;

        // Dtype — base marker trait for all types that can flow through the system
        pub trait Dtype: Copy + Clone {}

        // Num — numeric scalars; BITS is used for device buffer allocation
        pub trait Num: Dtype {
            const BITS: u8;
        }

        pub trait Float: Num {}
        pub trait Int: Num {}
        pub trait Bool: Dtype + Copy {}

        // Floating-point specialisations
        pub trait F8E4M3FN: Float {}
        pub trait F8E4M3FNUZ: Float {}
        pub trait F8E5M2: Float {}
        pub trait F8E5M2FNUZ: Float {}
        pub trait BF16: Float {}

        // Integer specialisations
        pub trait I4: Int {}

        // Primitive impls
        impl Dtype for bool {}

        impl Dtype for i8 {}
        impl Num for i8 {
            const BITS: u8 = 8;
        }
        impl Int for i8 {}

        impl Dtype for i16 {}
        impl Num for i16 {
            const BITS: u8 = 16;
        }
        impl Int for i16 {}

        impl Dtype for i32 {}
        impl Num for i32 {
            const BITS: u8 = 32;
        }
        impl Int for i32 {}

        impl Dtype for i64 {}
        impl Num for i64 {
            const BITS: u8 = 64;
        }
        impl Int for i64 {}

        impl Dtype for u8 {}
        impl Num for u8 {
            const BITS: u8 = 8;
        }
        impl Int for u8 {}

        impl Dtype for u16 {}
        impl Num for u16 {
            const BITS: u8 = 16;
        }
        impl Int for u16 {}

        impl Dtype for u32 {}
        impl Num for u32 {
            const BITS: u8 = 32;
        }
        impl Int for u32 {}

        impl Dtype for u64 {}
        impl Num for u64 {
            const BITS: u8 = 64;
        }
        impl Int for u64 {}

        impl Dtype for f32 {}
        impl Num for f32 {
            const BITS: u8 = 32;
        }
        impl Float for f32 {}

        impl Dtype for f64 {}
        impl Num for f64 {
            const BITS: u8 = 64;
        }
        impl Float for f64 {}

        // Tensor
        pub trait RankedTensor<D: Dtype, const RANK: usize>: Clone {
            const SHAPE: [usize; RANK];
        }

        pub trait Tensor<D: Dtype, const RANK: usize>: RankedTensor<D, RANK> {}

        /// Marker trait for eager (non-symbolic) tensors.
        ///
        /// Implement this on any tensor type that computes eagerly. The generic
        /// `Layer<T>` impls (Relu, Linear, Softmax, …) are gated on this marker so
        /// that the specific `Layer<SymTensor>` impls in `nn::graph` don't conflict.
        pub trait EagerTensor {}

        pub trait BoolTensor<const RANK: usize>: Tensor<bool, RANK> {}

        pub trait Comparison<I: Num, const RANK: usize> {
            type BoolTensor: BoolTensor<RANK>;

            fn lt(self, other: I) -> Self::BoolTensor;
        }

        pub trait I32Tensor<const RANK: usize>:
            Tensor<i32, RANK> + Add<i32> + Comparison<i32, RANK>
        {
        }

        // Offsets trait for adding tensor offsets to pointers
        pub trait AddOffsets<I: Int, const RANK: usize, T: Tensor<I, RANK>> {
            type Output;

            fn add_offsets(self, offsets: T) -> Self::Output;
        }

        // Pointer — Dtype itself (can be stored in tensors), no BITS needed
        pub trait Pointer<D: Dtype, const RANK: usize>:
            Sized + Copy + Clone + Dtype + AddOffsets<i32, RANK, Self::I32Tensor> + Add<Self>
        {
            type I32Tensor: I32Tensor<RANK>;
        }
    }
}
pub use triton::*;
fn kitchen_sink<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
) {
    fn combine_num<TT: Triton, DD: Dtype>(
        lhs: TT::Tensor<DD>,
        rhs: TT::Tensor<DD>,
    ) -> TT::Tensor<DD> {
        lhs + rhs
    }
    #[allow(clippy::empty_loop)]
    fn dummy_bool<TT: Triton>() -> TT::BoolTensor {
        loop {}
    }
    #[allow(clippy::empty_loop)]
    fn dummy_value<DD: Dtype>() -> DD {
        loop {}
    }

    // START HERE
    let _pid_x = T::program_id(Axis::X);
    let _pid_y = T::program_id(Axis::Y);
    let _pid_z = T::program_id(Axis::Z);
    let _nprog_x = T::num_programs(Axis::X);
    let r = T::arange(0, BLOCK_SIZE);
    let _ = r + n_elements;
    let _ = r - 1;
    let _ = r * 2;
    let z_shape = [BLOCK_SIZE];
    let z = T::zeros::<D>(unsafe { slice_from_raw_parts(&z_shape as *const i32, 1) });
    // if false {
    //     let _ = T::full::<D>(&[BLOCK_SIZE], dummy_value::<D>());
    // }
    let zl = T::zeros_like(z);
    let casted = T::cast::<D, D>(z, Some(FpDowncastRounding::Rtne), false);
    let _casted_rtz = T::cast::<D, D>(casted, Some(FpDowncastRounding::Rtz), true);
    let cat = T::cat(z, zl, true);
    let (ba, bb) = T::broadcast(cat, cat);
    let bto_shape = [BLOCK_SIZE];
    let bto = T::broadcast_to(ba, unsafe { slice_from_raw_parts(&bto_shape as *const i32, 1) });
    let ex = T::expand_dims(bto, 0);
    let perm0 = [0i32];
    let p = T::permute(ex, unsafe { slice_from_raw_parts(&perm0 as *const i32, 1) });
    let rs_shape = [BLOCK_SIZE];
    let rs = T::reshape(p, unsafe { slice_from_raw_parts(&rs_shape as *const i32, 1) }, false);
    let trans0 = [0i32];
    let tr = T::trans(rs, unsafe { slice_from_raw_parts(&trans0 as *const i32, 1) });
    let rv = T::ravel(tr, false);
    let vw_shape = [BLOCK_SIZE];
    let vw = T::view(rv, unsafe { slice_from_raw_parts(&vw_shape as *const i32, 1) });
    let jn = T::join(vw, bb);
    let il = T::interleave(jn, jn);
    let (sp0, sp1) = T::split(il);
    let dot = T::dot::<D, D>(sp0, sp1, None, Some(InputPrecision::TF32), Some(1));
    let _dot_tf32x3 = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::TF32x3), None);
    let _dot_ieee = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::IEEE), None);
    let scale = zl;
    let _dot_scaled = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E4M3,
        dot,
        scale,
        DotFormat::E5M2,
        None,
        true,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E2M1x2,
        dot,
        scale,
        DotFormat::E2M1x4,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::BF16x2,
        dot,
        scale,
        DotFormat::Int8,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::UInt8,
        dot,
        scale,
        DotFormat::E4M3,
        None,
        false,
    );
    let bp_shape = [BLOCK_SIZE]; let bp_strides = [1i32]; let bp_offsets = [0i32];
    let bp_bshape = [BLOCK_SIZE]; let bp_order = [0i32];
    let block_ptr = T::make_block_ptr(x_ptr,
        unsafe { slice_from_raw_parts(&bp_shape as *const i32, 1) },
        unsafe { slice_from_raw_parts(&bp_strides as *const i32, 1) },
        unsafe { slice_from_raw_parts(&bp_offsets as *const i32, 1) },
        unsafe { slice_from_raw_parts(&bp_bshape as *const i32, 1) },
        unsafe { slice_from_raw_parts(&bp_order as *const i32, 1) },
    );
    let adv_off = [1i32];
    let block_ptr2 = T::advance(block_ptr, unsafe { slice_from_raw_parts(&adv_off as *const i32, 1) });
    let td_shape = [BLOCK_SIZE]; let td_strides = [1i32]; let td_bshape = [BLOCK_SIZE];
    let tdesc = T::make_tensor_descriptor(
        y_ptr,
        unsafe { slice_from_raw_parts(&td_shape as *const i32, 1) },
        unsafe { slice_from_raw_parts(&td_strides as *const i32, 1) },
        unsafe { slice_from_raw_parts(&td_bshape as *const i32, 1) },
        Some(PaddingOption::Zero),
    );
    let tdn_shape = [BLOCK_SIZE]; let tdn_strides = [1i32]; let tdn_bshape = [BLOCK_SIZE];
    let tdesc_nan = T::make_tensor_descriptor(
        y_ptr,
        unsafe { slice_from_raw_parts(&tdn_shape as *const i32, 1) },
        unsafe { slice_from_raw_parts(&tdn_strides as *const i32, 1) },
        unsafe { slice_from_raw_parts(&tdn_bshape as *const i32, 1) },
        Some(PaddingOption::Nan),
    );
    let tdl_off = [0i32]; let tds_off = [0i32];
    let tdv = T::load_tensor_descriptor(tdesc, unsafe { slice_from_raw_parts(&tdl_off as *const i32, 1) });
    T::store_tensor_descriptor(tdesc_nan, unsafe { slice_from_raw_parts(&tds_off as *const i32, 1) }, tdv);
    let ptrs_shape = [BLOCK_SIZE];
    let ptrs = T::zeros::<T::Pointer<D>>(unsafe { slice_from_raw_parts(&ptrs_shape as *const i32, 1) });
    let z0_shape = [BLOCK_SIZE];
    let loaded = T::load::<D, 1>(
        ptrs,
        None,
        Some(T::zeros::<D>(unsafe { slice_from_raw_parts(&z0_shape as *const i32, 1) })),
        &[0],
        Some(PaddingOption::Zero),
        Some(CacheModifier::Ca),
        Some(EvictionPolicy::EvictFirst),
        false,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        Some(PaddingOption::Nan),
        Some(CacheModifier::Cg),
        Some(EvictionPolicy::EvictLast),
        true,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        None,
        Some(CacheModifier::Cv),
        Some(EvictionPolicy::NoEvict),
        false,
    );
    T::store::<D, 1>(
        ptrs,
        loaded,
        None,
        &[0],
        Some(CacheModifier::Wb),
        Some(EvictionPolicy::NoEvict),
    );
    T::store::<D, 1>(ptrs, loaded, None, &[0], Some(CacheModifier::Cs), None);
    if false {
        let cond: T::BoolTensor = dummy_bool::<T>();
        let _ = T::where_(cond, loaded, loaded);
        T::assume(cond);
        T::device_assert(cond, "kitchen_sink", Some(cond));
    }
    let fl = T::flip(loaded, Some(0));
    let _ = T::gather(fl, r, 0);
    let _abs = T::abs(loaded);
    let ceil = T::ceil(loaded);
    let floor = T::floor(ceil);
    let cos = T::cos(floor);
    let sin = T::sin(cos);
    let exp = T::exp(sin);
    let exp2 = T::exp2(exp);
    let log = T::log(exp2);
    let log2 = T::log2(log);
    let rsqrt = T::rsqrt(log2);
    let sig = T::sigmoid(rsqrt);
    let sqrt = T::sqrt(sig);
    let sqrt_rn = T::sqrt_rn(sqrt);
    let erf = T::erf(sqrt_rn);
    let smax = T::softmax(erf, Some(0), true, true);
    let mx = T::maximum(smax, smax);
    let mn = T::minimum(mx, smax);
    let cl = T::clamp(mn, smax, mx);
    let fm = T::fma(cl, smax, mx);
    let fd = T::fdiv(fm, smax, true);
    let dr = T::div_rn(fd, smax);
    // let _cd = T::cdiv(n_elements, BLOCK_SIZE);
    // let _swz = T::swizzle2d(0, 0, BLOCK_SIZE, BLOCK_SIZE, 1);
    // let _sum = T::sum(dr, Some(0), true);
    // let _max = T::max(dr, None, false);
    // let (_maxv, _maxi) = T::max_with_indices(dr, 0, true, false);
    // let _min = T::min(dr, Some(0), false);
    // let (_minv, _mini) = T::min_with_indices(dr, 0, true, false);
    // let _argmax = T::argmax(dr, 0, true, false);
    // let _argmin = T::argmin(dr, 0, true, false);
    // let _xors = T::xor_sum(T::zeros::<i32>(&[BLOCK_SIZE]), Some(0), false);
    // let _cumsum = T::cumsum(dr, 0, false);
    // let _cumprod = T::cumprod(dr, 0, true);
    // let _sort = T::sort(dr, Some(0), true);
    // let _hist = T::histogram(r, BLOCK_SIZE, None);
    // let _reduced = T::reduce::<D, D>(dr, 0, combine_num::<T, D>, false);
    // let _scan = T::associative_scan::<D>(dr, 0, combine_num::<T, D>, true);
    // let aptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    // let _ = T::atomic_add(aptrs, dr, None, Some(MemSem::Relaxed), Some(MemScope::Cta));
    // let _ = T::atomic_max(aptrs, dr, None, Some(MemSem::Acquire), Some(MemScope::Gpu));
    // let _ = T::atomic_min(aptrs, dr, None, Some(MemSem::Release), Some(MemScope::Sys));
    // let _ = T::atomic_xchg(aptrs, dr, None, Some(MemSem::AcqRel), Some(MemScope::Gpu));
    // let _ = T::atomic_cas(aptrs, dr, dr, Some(MemSem::AcqRel), Some(MemScope::Gpu));
    // let iptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    // let ival = T::zeros::<i32>(&[BLOCK_SIZE]);
    // let _ = T::atomic_and(iptrs, ival, None, Some(MemSem::Relaxed), Some(MemScope::Cta));
    // let _ = T::atomic_or(iptrs, ival, None, Some(MemSem::Acquire), Some(MemScope::Gpu));
    // let _ = T::atomic_xor(iptrs, ival, None, Some(MemSem::Release), Some(MemScope::Sys));
    // let u32a = T::zeros::<u32>(&[BLOCK_SIZE]);
    // let u32b = T::zeros::<u32>(&[BLOCK_SIZE]);
    // let _umulhi = T::umulhi(u32a, u32b);
    // let _rand = T::rand(123, r, 10);
    // let _randn = T::randn(123, r, 10);
    // let _randi = T::randint(123, r, 10);
    // let _rand4 = T::randint4x(123, r, 10);
    // let _asm = T::inline_asm_elementwise::<D>("", "", true, 1);
    // let mo = T::multiple_of(dr, &[1]);
    // let mc = T::max_contiguous(mo, &[1]);
    // let mconst = T::max_constancy(mc, &[1]);
    // T::debug_barrier();
    // T::device_print("val=", mconst, false);
    // T::static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
    // T::static_print("kitchen_sink");
    // let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    // let _ = T::make_block_ptr(output_ptr, &[BLOCK_SIZE], &[1], &[0], &[BLOCK_SIZE], &[0]);
    // T::store::<D, 1>(out_ptrs, mconst, None, &[0], None, None);
    // let _ = block_ptr2;
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::Pointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(
    x_ptr: *mut f32,
    y_ptr: *mut f32,
    output_ptr: *mut f32,
    n_elements: i32,
) {
    let x_ptr = Pointer(x_ptr as *mut _);
    let y_ptr = Pointer(y_ptr as *mut _);
    let output_ptr = Pointer(output_ptr as *mut _);
    kitchen_sink::<LlvmTriton, f32, 1024>(x_ptr, y_ptr, output_ptr, n_elements);
}
