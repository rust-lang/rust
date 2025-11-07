#![allow(unused_unsafe)]

mod sve;
mod sve2;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub mod types;

// 导出辅助函数
#[inline(always)]
pub(crate) unsafe fn simd_reinterpret<T, U>(x: T) -> U {
    crate::intrinsics::simd::simd_cast(x)
}

#[inline(always)]
pub(crate) unsafe fn simd_cast<T, U>(x: T) -> U {
    crate::intrinsics::simd::simd_cast(x)
}

#[inline(always)]
pub(crate) unsafe fn simd_select<M, T>(m: M, a: T, b: T) -> T {
    crate::intrinsics::simd::simd_select(m, a, b)
}

// 标量类型转换 Trait（用于生成代码中的类型转换）
trait ScalarConversion: Sized {
    type Unsigned;
    type Signed;
    fn as_unsigned(self) -> Self::Unsigned;
    fn as_signed(self) -> Self::Signed;
}

// 基本整数类型实现
impl ScalarConversion for i8 {
    type Unsigned = u8;
    type Signed = i8;
    #[inline(always)]
    fn as_unsigned(self) -> u8 { self as u8 }
    #[inline(always)]
    fn as_signed(self) -> i8 { self }
}

impl ScalarConversion for u8 {
    type Unsigned = u8;
    type Signed = i8;
    #[inline(always)]
    fn as_unsigned(self) -> u8 { self }
    #[inline(always)]
    fn as_signed(self) -> i8 { self as i8 }
}

impl ScalarConversion for i16 {
    type Unsigned = u16;
    type Signed = i16;
    #[inline(always)]
    fn as_unsigned(self) -> u16 { self as u16 }
    #[inline(always)]
    fn as_signed(self) -> i16 { self }
}

impl ScalarConversion for u16 {
    type Unsigned = u16;
    type Signed = i16;
    #[inline(always)]
    fn as_unsigned(self) -> u16 { self }
    #[inline(always)]
    fn as_signed(self) -> i16 { self as i16 }
}

impl ScalarConversion for i32 {
    type Unsigned = u32;
    type Signed = i32;
    #[inline(always)]
    fn as_unsigned(self) -> u32 { self as u32 }
    #[inline(always)]
    fn as_signed(self) -> i32 { self }
}

impl ScalarConversion for u32 {
    type Unsigned = u32;
    type Signed = i32;
    #[inline(always)]
    fn as_unsigned(self) -> u32 { self }
    #[inline(always)]
    fn as_signed(self) -> i32 { self as i32 }
}

impl ScalarConversion for i64 {
    type Unsigned = u64;
    type Signed = i64;
    #[inline(always)]
    fn as_unsigned(self) -> u64 { self as u64 }
    #[inline(always)]
    fn as_signed(self) -> i64 { self }
}

impl ScalarConversion for u64 {
    type Unsigned = u64;
    type Signed = i64;
    #[inline(always)]
    fn as_unsigned(self) -> u64 { self }
    #[inline(always)]
    fn as_signed(self) -> i64 { self as i64 }
}

// 指针类型实现 - 分别为有符号和无符号指针实现
macro_rules! impl_scalar_conversion_for_ptr {
    ($(($unsigned:ty, $signed:ty)),*) => {$(
        impl ScalarConversion for *const $unsigned {
            type Unsigned = *const $unsigned;
            type Signed = *const $signed;
            #[inline(always)]
            fn as_unsigned(self) -> *const $unsigned { self }
            #[inline(always)]
            fn as_signed(self) -> *const $signed { self as *const $signed }
        }
        
        impl ScalarConversion for *const $signed {
            type Unsigned = *const $unsigned;
            type Signed = *const $signed;
            #[inline(always)]
            fn as_unsigned(self) -> *const $unsigned { self as *const $unsigned }
            #[inline(always)]
            fn as_signed(self) -> *const $signed { self }
        }
        
        impl ScalarConversion for *mut $unsigned {
            type Unsigned = *mut $unsigned;
            type Signed = *mut $signed;
            #[inline(always)]
            fn as_unsigned(self) -> *mut $unsigned { self }
            #[inline(always)]
            fn as_signed(self) -> *mut $signed { self as *mut $signed }
        }
        
        impl ScalarConversion for *mut $signed {
            type Unsigned = *mut $unsigned;
            type Signed = *mut $signed;
            #[inline(always)]
            fn as_unsigned(self) -> *mut $unsigned { self as *mut $unsigned }
            #[inline(always)]
            fn as_signed(self) -> *mut $signed { self }
        }
    )*};
}

impl_scalar_conversion_for_ptr!((u8, i8), (u16, i16), (u32, i32), (u64, i64));

// 导出所有类型和函数
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve::*;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve2::*;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use types::*;
