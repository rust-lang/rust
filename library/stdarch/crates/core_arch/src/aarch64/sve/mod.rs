#![allow(unused_unsafe)]

mod sve;
mod sve2;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub mod types;

// ================================
// 修复点 1/2：去掉 simd_*，改为位级转换
// ================================
#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_reinterpret<T, U>(x: T) -> U {
    // 纯位级重解释；SVE 封装类型在这层视为opaque，避免走 simd_cast 触发 E0511
    core::mem::transmute_copy::<T, U>(&x)
}

#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_cast<T, U>(x: T) -> U {
    // 多数 SVE "cast"在 stdarch 内部只是布局相同的重解释；按位转即可
    // 如需数值语义转换，请在具体 API 内对接相应 LLVM SVE convert 内建。
    core::mem::transmute_copy::<T, U>(&x)
}

// ================================
// 修复点 3/3：逐类型绑定 LLVM SVE `sel` 内建，替代 simd_select
//   说明：SVE 的“按谓词选择”在 LLVM 里是 aarch64.sve.sel.* 内建，
//        名字与元素类型/宽度对应，如：nxv16i8/nxv8i16/nxv4i32/nxv2i64、nxv4f32/nxv2f64。
//        这是最稳妥的做法，避免把非SIMD类型喂给 simd_select 触发 E0511。 
// ================================
use types::*;

// 用 trait 把选择操作"静态分派"到对应的 LLVM SVE sel 内建上
pub(crate) trait __SveSelect {
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self;
}

// 声明 LLVM 内建函数（每个唯一的后缀只声明一次）
// 使用泛型函数指针类型，避免重复声明
unsafe extern "C" {
    #[link_name = "llvm.aarch64.sve.sel.nxv16i8"]
    fn __llvm_sve_sel_nxv16i8(mask: svbool_t, a: svint8_t, b: svint8_t) -> svint8_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv8i16"]
    fn __llvm_sve_sel_nxv8i16(mask: svbool_t, a: svint16_t, b: svint16_t) -> svint16_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv4i32"]
    fn __llvm_sve_sel_nxv4i32(mask: svbool_t, a: svint32_t, b: svint32_t) -> svint32_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv2i64"]
    fn __llvm_sve_sel_nxv2i64(mask: svbool_t, a: svint64_t, b: svint64_t) -> svint64_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv4f32"]
    fn __llvm_sve_sel_nxv4f32(mask: svbool_t, a: svfloat32_t, b: svfloat32_t) -> svfloat32_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv2f64"]
    fn __llvm_sve_sel_nxv2f64(mask: svbool_t, a: svfloat64_t, b: svfloat64_t) -> svfloat64_t;
    
    #[link_name = "llvm.aarch64.sve.sel.nxv16i1"]
    fn __llvm_sve_sel_nxv16i1(mask: svbool_t, a: svbool_t, b: svbool_t) -> svbool_t;
}

// 为每个类型实现 trait，调用对应的 LLVM 内建函数
// 注意：svint8_t 和 svuint8_t 共享同一个 LLVM 内建函数（都是 nxv16i8）
// 由于它们在 LLVM 层面是相同的类型，可以直接使用 transmute 进行类型转换
impl __SveSelect for svint8_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv16i8(mask, a, b)
    }
}

impl __SveSelect for svuint8_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        // svuint8_t 和 svint8_t 在 LLVM 层面是相同的类型（都是 nxv16i8）
        core::mem::transmute(__llvm_sve_sel_nxv16i8(mask, core::mem::transmute(a), core::mem::transmute(b)))
    }
}

impl __SveSelect for svint16_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv8i16(mask, a, b)
    }
}

impl __SveSelect for svuint16_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv8i16(mask, core::mem::transmute(a), core::mem::transmute(b)))
    }
}

impl __SveSelect for svint32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv4i32(mask, a, b)
    }
}

impl __SveSelect for svuint32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv4i32(mask, core::mem::transmute(a), core::mem::transmute(b)))
    }
}

impl __SveSelect for svint64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv2i64(mask, a, b)
    }
}

impl __SveSelect for svuint64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        core::mem::transmute(__llvm_sve_sel_nxv2i64(mask, core::mem::transmute(a), core::mem::transmute(b)))
    }
}

impl __SveSelect for svfloat32_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv4f32(mask, a, b)
    }
}

impl __SveSelect for svfloat64_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv2f64(mask, a, b)
    }
}

// svbool_t 是 1 位谓词向量，对应 nxv16i1
impl __SveSelect for svbool_t {
    #[inline(always)]
    unsafe fn sel(mask: svbool_t, a: Self, b: Self) -> Self {
        __llvm_sve_sel_nxv16i1(mask, a, b)
    }
}

// 如果你在 types.rs 支持了 f16 / bf16 / mfloat8，可按需解开/补齐：
// impl_sve_select!("nxv8f16",  svfloat16_t);
// impl_sve_select!("nxv8bf16", svbfloat16_t);
// impl_sve_select!("nxv16f8",  svmfloat8_t);

// 实现从不同宽度的谓词类型到 svbool_t 的转换
// 注意：这些实现直接使用 transmute_copy，不需要 target feature
// 因为 transmute_copy 是纯位级转换，不涉及 SVE 指令
impl From<svbool2_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool2_t) -> Self {
        // 使用 transmute_copy 进行位级转换，不需要 target feature
        unsafe { core::mem::transmute_copy(&x) }
    }
}

impl From<svbool4_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool4_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

impl From<svbool8_t> for svbool_t {
    #[inline(always)]
    fn from(x: svbool8_t) -> Self {
        unsafe { core::mem::transmute_copy(&x) }
    }
}

// 公开的"选择"总入口：保持原函数签名不变（被 sve/*.rs 调用）
// 现在它不再走 simd_select，而是经 trait 静态分派到 LLVM SVE `sel`
#[inline]
#[target_feature(enable = "sve")]
pub(crate) unsafe fn simd_select<M, T>(m: M, a: T, b: T) -> T
where
    // SVE 谓词统一为 svbool_t；避免出现 svbool4_t/svbool8_t 这类"假类型"
    M: Into<svbool_t>,
    T: __SveSelect,
{
    let mask: svbool_t = m.into();
    <T as __SveSelect>::sel(mask, a, b)
}

// -------- 下面保持你原有的标量转换 Trait 实现不变 --------
trait ScalarConversion: Sized {
    type Unsigned;
    type Signed;
    fn as_unsigned(self) -> Self::Unsigned;
    fn as_signed(self) -> Self::Signed;
}

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

// 指针类型实现
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

// 维持对外导出
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve::*;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use sve2::*;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use types::*;

// a) 外部内建
unsafe extern "C" {
    #[link_name = "llvm.aarch64.sve.cntw"]
    fn __llvm_sve_cntw() -> i32;

    #[link_name = "llvm.aarch64.sve.whilelt"]
    fn __llvm_sve_whilelt_i32(i: i32, n: i32) -> svbool_t;
}

// b) 对外 API
#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn svcntw() -> i32 { __llvm_sve_cntw() }

#[inline]
#[target_feature(enable = "sve")]
pub unsafe fn svwhilelt_b32(i: i32, n: i32) -> svbool_t {
    __llvm_sve_whilelt_i32(i, n)
}
