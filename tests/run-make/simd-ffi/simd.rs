// ensures that public symbols are not removed completely
#![crate_type = "lib"]
// we can compile to a variety of platforms, because we don't need
// cross-compiled standard libraries.
#![feature(no_core, auto_traits)]
#![no_core]
#![feature(repr_simd, simd_ffi, link_llvm_intrinsics, lang_items, rustc_attrs)]

#[derive(Copy)]
#[repr(simd)]
pub struct f32x4([f32; 4]);

extern "C" {
    #[link_name = "llvm.sqrt.v4f32"]
    fn vsqrt(x: f32x4) -> f32x4;
}

pub fn foo(x: f32x4) -> f32x4 {
    unsafe { vsqrt(x) }
}

#[derive(Copy)]
#[repr(simd)]
pub struct i32x4([i32; 4]);

extern "C" {
    // _mm_sll_epi32
    #[cfg(all(any(target_arch = "x86", target_arch = "x86-64"), target_feature = "sse2"))]
    #[link_name = "llvm.x86.sse2.psll.d"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;

    // vmaxq_s32
    #[cfg(target_arch = "arm")]
    #[link_name = "llvm.arm.neon.vmaxs.v4i32"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;
    // vmaxq_s32
    #[cfg(target_arch = "aarch64")]
    #[link_name = "llvm.aarch64.neon.maxs.v4i32"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;

    // Use a generic LLVM intrinsic to do type checking on other platforms
    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86-64"), target_feature = "sse2"),
        target_arch = "arm",
        target_arch = "aarch64"
    )))]
    #[link_name = "llvm.smax.v4i32"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;
}

pub fn bar(a: i32x4, b: i32x4) -> i32x4 {
    unsafe { integer(a, b) }
}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "copy"]
pub trait Copy {}

impl Copy for f32 {}
impl Copy for i32 {}
impl Copy for [f32; 4] {}
impl Copy for [i32; 4] {}

pub mod marker {
    pub use Copy;
}

#[lang = "freeze"]
auto trait Freeze {}

#[macro_export]
#[rustc_builtin_macro]
macro_rules! Copy {
    () => {};
}
#[macro_export]
#[rustc_builtin_macro]
macro_rules! derive {
    () => {};
}
