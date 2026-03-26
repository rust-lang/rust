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

pub mod std {
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

    use std::ops::Add;

    use self::types::{self as ty};

    pub use types::*;

    #[repr(i32)]
    pub enum Axis {
        X = 0,
        Y = 1,
        Z = 2,
    }

    pub trait Triton
    where
        Self::I32Tensor: Add<i32, Output = Self::I32Tensor>,
        Self::I32Tensor: Comparison<i32, BoolTensor = Self::BoolTensor>,
    {
        type BF16: ty::BF16;
        type BoolTensor: ty::BoolTensor;
        type I32Tensor: ty::I32Tensor;
        type Tensor<D: ty::Dtype>: ty::Tensor<D> + Add<Self::Tensor<D>, Output = Self::Tensor<D>>;
        type Pointer<D: ty::Dtype>: ty::Pointer<D, I32Tensor = Self::I32Tensor>
            + AddOffsets<i32, Self::I32Tensor, Output = Self::Tensor<Self::Pointer<D>>>;

        fn program_id(axis: Axis) -> i32;

        fn num_programs(axis: Axis) -> i32;

        fn arange(start: impl Into<i32>, end: impl Into<i32>) -> Self::I32Tensor;

        fn load<D: ty::Dtype>(
            ptr: Self::Tensor<Self::Pointer<D>>,
            mask: Self::BoolTensor,
        ) -> Self::Tensor<D>;

        fn store<D: ty::Dtype>(
            dest: Self::Tensor<Self::Pointer<D>>,
            src: Self::Tensor<D>,
            mask: Self::BoolTensor,
        );
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

        use std::ops::{Add, Mul};

        // Dtype Type
        pub trait Dtype: Copy + Clone {}

        pub trait Num: Dtype {}

        pub trait Float: Num {}
        pub trait Int: Num {}
        pub trait Bool: Dtype + Copy {}

        // Tensor
        pub trait RankedTensor<D: Dtype>: Copy + Clone {}

        // Floating-point types
        pub trait F8E4M3FN: Float {}
        pub trait F8E4M3FNUZ: Float {}
        pub trait F8E5M2: Float {}
        pub trait F8E5M2FNUZ: Float {}

        pub trait BF16: Float {}

        pub trait I4: Int {}

        impl Dtype for bool {}

        impl Dtype for i8 {}
        impl Num for i8 {}
        impl Int for i8 {}

        impl Dtype for i16 {}
        impl Num for i16 {}
        impl Int for i16 {}

        impl Dtype for i32 {}
        impl Num for i32 {}
        impl Int for i32 {}

        impl Dtype for i64 {}
        impl Num for i64 {}
        impl Int for i64 {}

        impl Dtype for u8 {}
        impl Num for u8 {}
        impl Int for u8 {}

        impl Dtype for u16 {}
        impl Num for u16 {}
        impl Int for u16 {}

        impl Dtype for u32 {}
        impl Num for u32 {}
        impl Int for u32 {}

        impl Dtype for u64 {}
        impl Num for u64 {}
        impl Int for u64 {}

        impl Dtype for f32 {}
        impl Num for f32 {}
        impl Float for f32 {}

        impl Dtype for f64 {}
        impl Num for f64 {}
        impl Float for f64 {}

        // Int Tensor
        pub trait Tensor<D: Dtype>: RankedTensor<D> {}

        pub trait BoolTensor: Tensor<bool> {}

        pub trait Comparison<I: Num> {
            type BoolTensor: BoolTensor;

            fn lt(self, other: I) -> Self::BoolTensor;
        }
        pub trait I32Tensor: Tensor<i32> + Add<i32> + Comparison<i32> {}

        // Offsets trait for adding tensor offsets to pointers
        pub trait AddOffsets<I: Int, T: Tensor<I>> {
            type Output;

            fn add_offsets(self, offsets: T) -> Self::Output;
        }

        // Pointer Type
        pub trait Pointer<D: Dtype>:
            Sized + Copy + Clone + Dtype + AddOffsets<i32, Self::I32Tensor> + Add<Self>
        {
            type I32Tensor: I32Tensor;
        }
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
            use super::super::{Axis, types as ty};

            pub struct LlvmTriton {}

            impl Triton for LlvmTriton {
                type BF16 = num::BF16;

                type BoolTensor = tensor::BoolTensor;
                type I32Tensor = tensor::I32Tensor;
                type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
                type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

                #[inline(never)]
                fn program_id(_axis: Axis) -> i32 {
                    // dummy implementation not used in final output
                    0
                }

                #[inline(never)]
                fn num_programs(_axis: Axis) -> i32 {
                    // dummy implementation not used in final output
                    0
                }

                #[inline(never)]
                fn arange(_start: impl Into<i32>, _end: impl Into<i32>) -> Self::I32Tensor {
                    // dummy implementation not used in final output
                    tensor::Tensor(0 as *mut i32)
                }

                #[inline(never)]
                fn load<D: ty::Dtype>(
                    _ptr: Self::Tensor<Self::Pointer<D>>,
                    _mask: Self::BoolTensor,
                ) -> Self::Tensor<D> {
                    // dummy implementation not used in final output
                    tensor::Tensor(0 as *mut D)
                }

                #[inline(never)]
                fn store<D: ty::Dtype>(
                    _dest: Self::Tensor<Self::Pointer<D>>,
                    _src: Self::Tensor<D>,
                    _mask: Self::BoolTensor,
                ) {
                    // nop
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

                use std::ops::Add;

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

                impl<D: ty::Dtype> ty::Pointer<D> for Pointer<D> {
                    type I32Tensor = I32Tensor;
                }

                // Implement AddOffsets for I64Tensor
                impl<D: ty::Dtype> ty::AddOffsets<i32, I32Tensor> for Pointer<D> {
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
                impl ty::Num for BF16 {}
                impl ty::Float for BF16 {}
                impl ty::BF16 for BF16 {}
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

                use std::ops::Add;

                use super::super::super::types::{self as ty};

                /*--------------------------------- Tensor ---------------------------------*/

                pub struct Tensor<D: ty::Dtype>(pub *mut D);
                impl<D: ty::Dtype> Clone for Tensor<D> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
                impl<D: ty::Dtype> Copy for Tensor<D> {}

                impl<D: ty::Dtype> ty::Tensor<D> for Tensor<D> {}
                impl<D: ty::Dtype> ty::RankedTensor<D> for Tensor<D> {}

                // Element-wise addition for tensors
                impl<D: ty::Dtype> Add<Tensor<D>> for Tensor<D> {
                    type Output = Tensor<D>;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: Tensor<D>) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut D)
                    }
                }

                pub type BoolTensor = Tensor<bool>;
                impl ty::BoolTensor for BoolTensor {}

                pub type I32Tensor = Tensor<i32>;

                impl ty::I32Tensor for I32Tensor {}

                impl ty::Comparison<i32> for I32Tensor {
                    type BoolTensor = BoolTensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn lt(self, _other: i32) -> Self::BoolTensor {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut bool)
                    }
                }

                // Blanket implementation for any type implementing I64, including <I32 as Mul<u32>>::Output
                impl Add<i32> for I32Tensor {
                    type Output = I32Tensor;

                    #[inline(never)]
                    #[allow(clippy::zero_ptr)]
                    fn add(self, _rhs: i32) -> Self::Output {
                        // dummy implementation not used in final output
                        Tensor(0 as *mut i32)
                    }
                }
            }
        }
    }
}
pub use triton::*;

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

    tensor_add::<LlvmTriton, f32, 128>(x_ptr, y_ptr, output_ptr, n_elements);
}
pub extern "C" fn tensor_add<T: Triton, D: types::Dtype, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
) {
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let offsets = T::arange(0, BLOCK_SIZE) + block_start;
    let mask = offsets.lt(n_elements);
    let x = T::load(x_ptr.add_offsets(offsets), mask);
    let y = T::load(y_ptr.add_offsets(offsets), mask);
    let output = x + y;
    T::store(output_ptr.add_offsets(offsets), output, mask);
}
