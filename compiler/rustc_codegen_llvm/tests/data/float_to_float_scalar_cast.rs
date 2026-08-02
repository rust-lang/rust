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
#![feature(intrinsics, lang_items)]
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

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {}

// Enable &[T; N] → &[T] coercions without unsafe slice_from_raw_parts.
// The compiler automatically implements Unsize<[T]> for [T; N]; this impl
// wires up the syntactic coercion for shared references.
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}

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

#[lang = "Option"]
pub enum Option<T> {
    #[lang = "None"]
    None,
    #[lang = "Some"]
    Some(T),
}

use Option::*;

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

        #[lang = "mul_assign"]
        pub trait MulAssign<RHS = Self> {
            fn mul_assign(&mut self, rhs: RHS);
        }

        impl MulAssign for i32 {
            fn mul_assign(&mut self, rhs: i32) {}
        }
        impl MulAssign for i64 {
            fn mul_assign(&mut self, rhs: i64) {}
        }
        impl MulAssign for f32 {
            fn mul_assign(&mut self, rhs: f32) {}
        }
        impl MulAssign for f64 {
            fn mul_assign(&mut self, rhs: f64) {}
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

        #[lang = "add_assign"]
        pub trait AddAssign<RHS = Self> {
            fn add_assign(&mut self, rhs: RHS);
        }

        impl AddAssign for i32 {
            fn add_assign(&mut self, rhs: i32) {}
        }
        impl AddAssign for u32 {
            fn add_assign(&mut self, rhs: u32) {}
        }
        impl AddAssign for u64 {
            fn add_assign(&mut self, rhs: u64) {}
        }
        impl AddAssign for i64 {
            fn add_assign(&mut self, rhs: i64) {}
        }
        impl AddAssign for f32 {
            fn add_assign(&mut self, rhs: f32) {}
        }
        impl AddAssign for f64 {
            fn add_assign(&mut self, rhs: f64) {}
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

        #[lang = "sub_assign"]
        pub trait SubAssign<RHS = Self> {
            fn sub_assign(&mut self, rhs: RHS);
        }

        impl SubAssign for i32 {
            fn sub_assign(&mut self, rhs: i32) {}
        }
        impl SubAssign for u32 {
            fn sub_assign(&mut self, rhs: u32) {}
        }
        impl SubAssign for i64 {
            fn sub_assign(&mut self, rhs: i64) {}
        }
        impl SubAssign for f32 {
            fn sub_assign(&mut self, rhs: f32) {}
        }
        impl SubAssign for f64 {
            fn sub_assign(&mut self, rhs: f64) {}
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

        #[lang = "div_assign"]
        pub trait DivAssign<RHS = Self> {
            fn div_assign(&mut self, rhs: RHS);
        }

        impl DivAssign for i32 {
            fn div_assign(&mut self, rhs: i32) {}
        }
        impl DivAssign for i64 {
            fn div_assign(&mut self, rhs: i64) {}
        }
        impl DivAssign for f32 {
            fn div_assign(&mut self, rhs: f32) {}
        }
        impl DivAssign for f64 {
            fn div_assign(&mut self, rhs: f64) {}
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

        #[lang = "Range"]
        pub struct Range<Idx> {
            pub start: Idx,
            pub end: Idx,
        }
    }
}

#[lang = "eq"]
pub trait PartialEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool;
}

impl PartialEq for i32 {
    fn eq(&self, _other: &i32) -> bool { false }
    fn ne(&self, _other: &i32) -> bool { false }
}

#[lang = "partial_ord"]
pub trait PartialOrd<Rhs: ?Sized = Self> {
    fn lt(&self, other: &Rhs) -> bool;
    fn gt(&self, other: &Rhs) -> bool;
    fn le(&self, other: &Rhs) -> bool;
    fn ge(&self, other: &Rhs) -> bool;
}

impl PartialOrd for i32 {
    fn lt(&self, _other: &i32) -> bool { false }
    fn gt(&self, _other: &i32) -> bool { false }
    fn le(&self, _other: &i32) -> bool { false }
    fn ge(&self, _other: &i32) -> bool { false }
}

pub mod iter {
    #[lang = "iterator"]
    pub trait Iterator {
        type Item;

        #[lang = "next"]
        fn next(&mut self) -> super::Option<Self::Item>;
    }

    pub trait IntoIterator {
        type Item;
        type IntoIter: Iterator<Item = Self::Item>;

        #[lang = "into_iter"]
        fn into_iter(self) -> Self::IntoIter;
    }

    impl Iterator for super::core::ops::Range<i32> {
        type Item = i32;

        fn next(&mut self) -> super::Option<Self::Item> {
            if self.start < self.end {
                let v = self.start;
                self.start = v + 1;
                super::Option::Some(v)
            } else {
                super::Option::None
            }
        }
    }

    impl IntoIterator for super::core::ops::Range<i32> {
        type Item = i32;
        type IntoIter = super::core::ops::Range<i32>;

        fn into_iter(self) -> Self::IntoIter {
            self
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

        /*------------------------------ Comparison Ops ------------------------------*/

        /// Element-wise less-than between two tensors.
        fn lt<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
        /// Element-wise less-than-or-equal between two tensors.
        fn le<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
        /// Element-wise greater-than between two tensors.
        fn gt<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
        /// Element-wise greater-than-or-equal between two tensors.
        fn ge<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
        /// Element-wise equality between two tensors.
        fn eq<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
        /// Element-wise inequality between two tensors.
        fn ne<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;

        /// Element-wise less-than against a scalar.
        fn lt_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
        /// Element-wise less-than-or-equal against a scalar.
        fn le_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
        /// Element-wise greater-than against a scalar.
        fn gt_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
        /// Element-wise greater-than-or-equal against a scalar.
        fn ge_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
        /// Element-wise equality against a scalar.
        fn eq_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
        /// Element-wise inequality against a scalar.
        fn ne_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;

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
                type BF16 = num::LlvmBF16;
                type BoolTensor = tensor::LlvmBoolTensor;
                type I32Tensor = tensor::LlvmI32Tensor;
                type Tensor<D: ty::Dtype> = tensor::LlvmTensor<D>;
                type Pointer<D: ty::Dtype> = pointer::LlvmPointer<D>;

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
                    tensor::LlvmTensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn zeros<D: ty::Dtype>(_shape: &[i32]) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn zeros_like<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn full<D: ty::Dtype>(_shape: &[i32], _value: D) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cast<Src: ty::Dtype, Dst: ty::Dtype>(
                    _x: Self::Tensor<Src>,
                    _fp_downcast_rounding: Option<FpDowncastRounding>,
                    _bitcast: bool,
                ) -> Self::Tensor<Dst> {
                    tensor::LlvmTensor(0 as *mut Dst)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cat<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                    _can_reorder: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                /*------------------------------ Shape Manipulation Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn broadcast<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                ) -> (Self::Tensor<D>, Self::Tensor<D>) {
                    let t = tensor::LlvmTensor(0 as *mut D);
                    (t, t)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn broadcast_to<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _shape: &[i32],
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn expand_dims<D: ty::Dtype>(_x: Self::Tensor<D>, _axis: i32) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn permute<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn reshape<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _shape: &[i32],
                    _can_reorder: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn trans<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ravel<D: ty::Dtype>(_x: Self::Tensor<D>, _can_reorder: bool) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn view<D: ty::Dtype>(_x: Self::Tensor<D>, _shape: &[i32]) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn join<D: ty::Dtype>(_a: Self::Tensor<D>, _b: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn interleave<D: ty::Dtype>(
                    _a: Self::Tensor<D>,
                    _b: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn split<D: ty::Dtype>(_x: Self::Tensor<D>) -> (Self::Tensor<D>, Self::Tensor<D>) {
                    let t = tensor::LlvmTensor(0 as *mut D);
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
                    tensor::LlvmTensor(0 as *mut O)
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
                    tensor::LlvmTensor(0 as *mut O)
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
                    pointer::LlvmPointer(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn advance<D: ty::Dtype>(
                    _ptr: Self::Pointer<D>,
                    _offsets: &[i32],
                ) -> Self::Pointer<D> {
                    pointer::LlvmPointer(0 as *mut D)
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
                    pointer::LlvmPointer(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn load_tensor_descriptor<D: ty::Dtype>(
                    _desc: Self::Pointer<D>,
                    _offsets: &[i32],
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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

                /*------------------------------ Comparison Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn lt<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn le<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn gt<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ge<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn eq<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ne<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn lt_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn le_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn gt_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ge_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn eq_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ne_scalar<D: ty::Num>(_x: Self::Tensor<D>, _y: D) -> Self::BoolTensor {
                    tensor::LlvmTensor(0 as *mut bool)
                }

                /*------------------------------ Indexing Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn where_<D: ty::Dtype>(
                    _cond: Self::BoolTensor,
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn flip<D: ty::Dtype>(_x: Self::Tensor<D>, _dim: Option<i32>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn gather<D: ty::Dtype>(
                    _src: Self::Tensor<D>,
                    _index: Self::I32Tensor,
                    _axis: i32,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                /*------------------------------ Math Ops — Unary (floating-point) ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn abs<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn ceil<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn floor<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cos<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sin<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn exp<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn exp2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn log<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn log2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn rsqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sigmoid<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sqrt_rn<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn erf<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn softmax<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _dim: Option<i32>,
                    _keep_dims: bool,
                    _ieee_rounding: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                /*------------------------------ Math Ops — Binary ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn maximum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn minimum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn clamp<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _lo: Self::Tensor<D>,
                    _hi: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn fma<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                    _z: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn fdiv<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                    _ieee_rounding: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn div_rn<D: ty::Float>(
                    _x: Self::Tensor<D>,
                    _y: Self::Tensor<D>,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn umulhi(_x: Self::Tensor<u32>, _y: Self::Tensor<u32>) -> Self::Tensor<u32> {
                    tensor::LlvmTensor(0 as *mut u32)
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
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn max<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn max_with_indices<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> (Self::Tensor<D>, Self::I32Tensor) {
                    (tensor::LlvmTensor(0 as *mut D), tensor::LlvmTensor(0 as *mut i32))
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn min<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn min_with_indices<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> (Self::Tensor<D>, Self::I32Tensor) {
                    (tensor::LlvmTensor(0 as *mut D), tensor::LlvmTensor(0 as *mut i32))
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn argmax<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> Self::I32Tensor {
                    tensor::LlvmTensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn argmin<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _tie_break_left: bool,
                    _keep_dims: bool,
                ) -> Self::I32Tensor {
                    tensor::LlvmTensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn xor_sum<D: ty::Int>(
                    _x: Self::Tensor<D>,
                    _axis: Option<i32>,
                    _keep_dims: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                /*------------------------------ Scan / Sort Ops ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cumsum<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn cumprod<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn sort<D: ty::Num>(
                    _x: Self::Tensor<D>,
                    _dim: Option<i32>,
                    _descending: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn histogram(
                    _x: Self::I32Tensor,
                    _num_bins: i32,
                    _mask: Option<Self::BoolTensor>,
                ) -> Self::I32Tensor {
                    tensor::LlvmTensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn reduce<D: ty::Dtype, O: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>,
                    _keep_dims: bool,
                ) -> Self::Tensor<O> {
                    tensor::LlvmTensor(0 as *mut O)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn associative_scan<D: ty::Dtype>(
                    _x: Self::Tensor<D>,
                    _axis: i32,
                    _combine_fn: fn(Self::Tensor<D>, Self::Tensor<D>) -> Self::Tensor<D>,
                    _reverse: bool,
                ) -> Self::Tensor<D> {
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
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
                    tensor::LlvmTensor(0 as *mut D)
                }

                /*------------------------------ Random Number Generation ------------------------------*/

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn rand(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::Tensor<f32> {
                    tensor::LlvmTensor(0 as *mut f32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randn(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::Tensor<f32> {
                    tensor::LlvmTensor(0 as *mut f32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randint(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> Self::I32Tensor {
                    tensor::LlvmTensor(0 as *mut i32)
                }

                #[inline(never)]
                #[allow(clippy::zero_ptr)]
                fn randint4x(
                    _seed: u32,
                    _offsets: Self::I32Tensor,
                    _n_rounds: i32,
                ) -> (Self::I32Tensor, Self::I32Tensor, Self::I32Tensor, Self::I32Tensor)
                {
                    let t = tensor::LlvmTensor(0 as *mut i32);
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
                    tensor::LlvmTensor(0 as *mut D)
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

                pub struct LlvmTensor<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for LlvmTensor<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for LlvmTensor<D> {}

                impl<D: ty::Dtype, const RANK: usize> ty::RankedTensor<D, RANK> for LlvmTensor<D> {
                    const SHAPE: [usize; RANK] = [0; RANK];
                }
                impl<D: ty::Dtype, const RANK: usize> ty::Tensor<D, RANK> for LlvmTensor<D> {}

                impl<D: ty::Dtype> Add<LlvmTensor<D>> for LlvmTensor<D> {
                    type Output = LlvmTensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: LlvmTensor<D>) -> Self::Output {
                        LlvmTensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Sub<LlvmTensor<D>> for LlvmTensor<D> {
                    type Output = LlvmTensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn sub(self, _rhs: LlvmTensor<D>) -> Self::Output {
                        LlvmTensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Mul<LlvmTensor<D>> for LlvmTensor<D> {
                    type Output = LlvmTensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn mul(self, _rhs: LlvmTensor<D>) -> Self::Output {
                        LlvmTensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Div<LlvmTensor<D>> for LlvmTensor<D> {
                    type Output = LlvmTensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn div(self, _rhs: LlvmTensor<D>) -> Self::Output {
                        LlvmTensor(0 as *mut D)
                    }
                }

                impl<D: ty::Dtype> Neg for LlvmTensor<D> {
                    type Output = LlvmTensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn neg(self) -> Self::Output {
                        LlvmTensor(0 as *mut D)
                    }
                }

                pub type LlvmBoolTensor = LlvmTensor<bool>;
                impl<const RANK: usize> ty::BoolTensor<RANK> for LlvmBoolTensor {}

                pub type LlvmI32Tensor = LlvmTensor<i32>;

                impl<const RANK: usize> ty::I32Tensor<RANK> for LlvmI32Tensor {}

                impl<D: ty::Num> ty::Comparison<D> for LlvmTensor<D> {
                    type BoolTensor = LlvmBoolTensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn lt(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn le(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn gt(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn ge(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn eq(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn ne(self, _other: D) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                }

                impl<D: ty::Num> ty::Comparison<LlvmTensor<D>> for LlvmTensor<D> {
                    type BoolTensor = LlvmBoolTensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn lt(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn le(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn gt(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn ge(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn eq(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn ne(self, _other: LlvmTensor<D>) -> Self::BoolTensor {
                        LlvmTensor(0 as *mut bool)
                    }
                }

                impl Add<i32> for LlvmI32Tensor {
                    type Output = LlvmI32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: i32) -> Self::Output {
                        LlvmTensor(0 as *mut i32)
                    }
                }

                impl Sub<i32> for LlvmI32Tensor {
                    type Output = LlvmI32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn sub(self, _rhs: i32) -> Self::Output {
                        LlvmTensor(0 as *mut i32)
                    }
                }

                impl Mul<i32> for LlvmI32Tensor {
                    type Output = LlvmI32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn mul(self, _rhs: i32) -> Self::Output {
                        LlvmTensor(0 as *mut i32)
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

                pub trait Float: Num {
                    fn from_f64(value: f64) -> Self;
                }
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
                impl Float for f32 {
                    fn from_f64(value: f64) -> Self {
                        value as f32
                    }
                }

                impl Dtype for f64 {}
                impl Num for f64 {
                    const BITS: u8 = 64;
                }
                impl Float for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

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

                pub trait Comparison<Rhs = Self> {
                    type BoolTensor;

                    fn lt(self, other: Rhs) -> Self::BoolTensor;
                    fn le(self, other: Rhs) -> Self::BoolTensor;
                    fn gt(self, other: Rhs) -> Self::BoolTensor;
                    fn ge(self, other: Rhs) -> Self::BoolTensor;
                    fn eq(self, other: Rhs) -> Self::BoolTensor;
                    fn ne(self, other: Rhs) -> Self::BoolTensor;
                }

                pub trait I32Tensor<const RANK: usize>:
                    Tensor<i32, RANK> + Add<i32> + Comparison<i32>
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

                pub struct LlvmBF16;
                impl Copy for LlvmBF16 {}
                impl Clone for LlvmBF16 {
                    #[inline(always)]
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                impl ty::Dtype for LlvmBF16 {}
                impl ty::Num for LlvmBF16 {
                    const BITS: u8 = 16;
                }
                impl ty::Float for LlvmBF16 {
                    fn from_f64(_value: f64) -> Self {
                        LlvmBF16
                    }
                }
                impl ty::BF16 for LlvmBF16 {}
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

                use crate::triton::llvm::triton::tensor::{LlvmI32Tensor, LlvmTensor};

                use super::super::super::types::{self as ty};

                pub struct LlvmPointer<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for LlvmPointer<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for LlvmPointer<D> {}

                impl<D: ty::Dtype> ty::Dtype for LlvmPointer<D> {}

                impl<D: ty::Dtype, const RANK: usize> ty::Pointer<D, RANK> for LlvmPointer<D> {
                    type I32Tensor = LlvmI32Tensor;
                }

                // Implement AddOffsets for Pointer
                impl<D: ty::Dtype, const RANK: usize> ty::AddOffsets<i32, RANK, LlvmI32Tensor> for LlvmPointer<D> {
                    type Output = LlvmTensor<Self>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add_offsets(self, _offsets: LlvmI32Tensor) -> Self::Output {
                        // dummy implementation not used in final output
                        LlvmTensor(0 as *mut Self)
                    }
                }

                impl<D: ty::Dtype> Add<LlvmPointer<D>> for LlvmPointer<D> {
                    type Output = Self;

                    #[inline(never)]
                    fn add(self, _other: LlvmPointer<D>) -> Self::Output {
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

        pub trait Float: Num {
            fn from_f64(value: f64) -> Self;
        }
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
        impl Float for f32 {
            fn from_f64(value: f64) -> Self {
                value as f32
            }
        }

        impl Dtype for f64 {}
        impl Num for f64 {
            const BITS: u8 = 64;
        }
        impl Float for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

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

        pub trait Comparison<Rhs = Self> {
            type BoolTensor;

            fn lt(self, other: Rhs) -> Self::BoolTensor;
            fn le(self, other: Rhs) -> Self::BoolTensor;
            fn gt(self, other: Rhs) -> Self::BoolTensor;
            fn ge(self, other: Rhs) -> Self::BoolTensor;
            fn eq(self, other: Rhs) -> Self::BoolTensor;
            fn ne(self, other: Rhs) -> Self::BoolTensor;
        }

        pub trait I32Tensor<const RANK: usize>:
            Tensor<i32, RANK> + Add<i32> + Comparison<i32>
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

// Regression test for: `codegen_cast`'s missing `CastKind::FloatToFloat` arm silently
// emitted `ub.poison` for scalar float widen/narrow casts (e.g. `x as f64`, `value as f32`),
// instead of a real `arith.extf`/`arith.truncf`. This is exactly the pattern
// `Float::from_f64(value: f64) -> Self` uses in vision-rs's generic Flash-Attention-2 /
// PSA kernels (`D::from_f64(x as f64)`), where the poisoned constant silently corrupted
// the online-softmax running max/scale, producing wrong (non-NaN, then eventually NaN)
// results with no compile error. See compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs.
//
// Kernel: materialize a [BLOCK] tensor filled with `D::from_f64(x as f64)` (round-tripping
// the f32 kernel argument through f64, exactly like the generic Float trait does) and store
// it. With the bug, this compiled to `ub.poison`, register-allocated on top of whatever
// value happened to share that register — not caught by the compiler, only by comparing
// runtime output against the expected input.
fn float_to_float_scalar_cast<T: Triton, D: Float, const BLOCK: i32>(x: f32) {
    let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let filled = T::full::<D>(&[BLOCK], D::from_f64(x as f64));
    T::store::<D, 1>(out_ptrs, filled, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x: f32) {
    float_to_float_scalar_cast::<LlvmTriton, f32, 64>(x);
}
