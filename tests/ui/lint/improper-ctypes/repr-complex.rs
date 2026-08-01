// Currently we only guarantee that repr(complex) on a Complex<{float}> is ABI-compatible.
#![feature(repr_complex)]
#![feature(f16, f128)]
#![deny(improper_ctypes)]
#![crate_type = "lib"]

#[repr(complex)]
pub struct ComplexF16(f16, f16);

#[repr(complex)]
pub struct ComplexF32(f32, f32);

#[repr(complex)]
pub struct ComplexF64(f64, f64);

#[repr(complex)]
pub struct ComplexF128(f128, f128);

#[repr(transparent)]
pub struct Wrap(f32);

#[repr(complex)]
pub struct ComplexWrap(Wrap, Wrap);

#[repr(transparent)]
pub struct WrapWrap(Wrap);

#[repr(complex)]
pub struct ComplexWrapWrap(WrapWrap, WrapWrap);

#[repr(complex)]
pub struct ComplexI32(i32, i32);

#[repr(complex)]
pub struct ComplexU64(u64, u64);

extern "C" {
    pub fn complex_f16(x: ComplexF16);
    pub fn complex_f32(x: ComplexF32);
    pub fn complex_f64(x: ComplexF64);
    pub fn complex_f128(x: ComplexF128);

    pub fn complex_wrap(x: ComplexWrap);
    pub fn complex_wrap_wrap(x: ComplexWrapWrap);

    pub fn complex_i32(x: ComplexI32); //~ ERROR `extern` block uses type `ComplexI32`
    pub fn complex_u64(x: ComplexU64); //~ ERROR `extern` block uses type `ComplexU64`
}
