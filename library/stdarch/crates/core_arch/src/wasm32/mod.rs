//! WASM32 intrinsics

#[cfg(test)]
use stdarch_test::assert_instr;

mod atomic;
#[unstable(feature = "stdarch_wasm_atomic_wait", issue = "77839")]
pub use self::atomic::*;

mod simd128;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use self::simd128::*;

mod relaxed_simd;
#[stable(feature = "stdarch_wasm_relaxed_simd", since = "1.82.0")]
pub use self::relaxed_simd::*;

mod memory;
#[stable(feature = "simd_wasm32", since = "1.33.0")]
pub use self::memory::*;

/// Generates the [`unreachable`] instruction, which causes an unconditional [trap].
///
/// This function is safe to call and immediately aborts the execution.
///
/// [`unreachable`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-control
/// [trap]: https://webassembly.github.io/spec/core/intro/overview.html#trap
#[cfg_attr(test, assert_instr(unreachable))]
#[inline]
#[stable(feature = "unreachable_wasm32", since = "1.37.0")]
pub fn unreachable() -> ! {
    crate::intrinsics::abort()
}

/// Generates the [`f32.ceil`] instruction, returning the smallest integer greater than or equal to `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f32::ceil()`].
///
/// [`std::f32::ceil()`]: https://doc.rust-lang.org/std/primitive.f32.html#method.ceil
/// [`f32.ceil`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f32.ceil))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f32_ceil(a: f32) -> f32 {
    unsafe { crate::intrinsics::ceilf32(a) }
}

/// Generates the [`f32.floor`] instruction, returning the largest integer less than or equal to `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f32::floor()`].
///
/// [`std::f32::floor()`]: https://doc.rust-lang.org/std/primitive.f32.html#method.floor
/// [`f32.floor`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f32.floor))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f32_floor(a: f32) -> f32 {
    unsafe { crate::intrinsics::floorf32(a) }
}

/// Generates the [`f32.trunc`] instruction, roundinging to the nearest integer towards zero.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f32::trunc()`].
///
/// [`std::f32::trunc()`]: https://doc.rust-lang.org/std/primitive.f32.html#method.trunc
/// [`f32.trunc`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f32.trunc))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f32_trunc(a: f32) -> f32 {
    unsafe { crate::intrinsics::truncf32(a) }
}

/// Generates the [`f32.nearest`] instruction, roundinging to the nearest integer. Rounds half-way
/// cases to the number with an even least significant digit.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f32::round_ties_even()`].
///
/// [`std::f32::round_ties_even()`]: https://doc.rust-lang.org/std/primitive.f32.html#method.round_ties_even
/// [`f32.nearest`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f32.nearest))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f32_nearest(a: f32) -> f32 {
    crate::intrinsics::round_ties_even_f32(a)
}

/// Generates the [`f32.sqrt`] instruction, returning the square root of the number `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f32::sqrt()`].
///
/// [`std::f32::sqrt()`]: https://doc.rust-lang.org/std/primitive.f32.html#method.sqrt
/// [`f32.sqrt`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f32.sqrt))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f32_sqrt(a: f32) -> f32 {
    unsafe { crate::intrinsics::sqrtf32(a) }
}

/// Generates the [`f64.ceil`] instruction, returning the smallest integer greater than or equal to `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f64::ceil()`].
///
/// [`std::f64::ceil()`]: https://doc.rust-lang.org/std/primitive.f64.html#method.ceil
/// [`f64.ceil`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f64.ceil))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f64_ceil(a: f64) -> f64 {
    unsafe { crate::intrinsics::ceilf64(a) }
}

/// Generates the [`f64.floor`] instruction, returning the largest integer less than or equal to `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f64::floor()`].
///
/// [`std::f64::floor()`]: https://doc.rust-lang.org/std/primitive.f64.html#method.floor
/// [`f64.floor`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f64.floor))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f64_floor(a: f64) -> f64 {
    unsafe { crate::intrinsics::floorf64(a) }
}

/// Generates the [`f64.trunc`] instruction, roundinging to the nearest integer towards zero.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f64::trunc()`].
///
/// [`std::f64::trunc()`]: https://doc.rust-lang.org/std/primitive.f64.html#method.trunc
/// [`f64.trunc`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f64.trunc))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f64_trunc(a: f64) -> f64 {
    unsafe { crate::intrinsics::truncf64(a) }
}

/// Generates the [`f64.nearest`] instruction, roundinging to the nearest integer. Rounds half-way
/// cases to the number with an even least significant digit.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f64::round_ties_even()`].
///
/// [`std::f64::round_ties_even()`]: https://doc.rust-lang.org/std/primitive.f64.html#method.round_ties_even
/// [`f64.nearest`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f64.nearest))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f64_nearest(a: f64) -> f64 {
    crate::intrinsics::round_ties_even_f64(a)
}

/// Generates the [`f64.sqrt`] instruction, returning the square root of the number `a`.
///
/// This method is useful when targeting `no_std` and is equivalent to [`std::f64::sqrt()`].
///
/// [`std::f64::sqrt()`]: https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt
/// [`f64.sqrt`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-numeric
#[cfg_attr(test, assert_instr(f64.sqrt))]
#[inline]
#[must_use = "method returns a new number and does not mutate the original value"]
#[unstable(feature = "wasm_numeric_instr", issue = "133908")]
pub fn f64_sqrt(a: f64) -> f64 {
    unsafe { crate::intrinsics::sqrtf64(a) }
}

unsafe extern "C-unwind" {
    #[link_name = "llvm.wasm.throw"]
    fn wasm_throw(tag: i32, ptr: *mut u8) -> !;
}

/// Generates the [`throw`] instruction from the [exception-handling proposal] for WASM.
///
/// This function is unlikely to be stabilized until codegen backends have better support.
///
/// [`throw`]: https://webassembly.github.io/exception-handling/core/syntax/instructions.html#syntax-instr-control
/// [exception-handling proposal]: https://github.com/WebAssembly/exception-handling
// FIXME: wasmtime does not currently support exception-handling, so cannot execute
//        a wasm module with the throw instruction in it. once it does, we can
//        reenable this attribute.
// #[cfg_attr(test, assert_instr(throw, TAG = 0, ptr = core::ptr::null_mut()))]
#[inline]
#[unstable(feature = "wasm_exception_handling_intrinsics", issue = "122465")]
pub unsafe fn throw<const TAG: i32>(ptr: *mut u8) -> ! {
    static_assert!(TAG == 0); // LLVM only supports tag 0 == C++ right now.
    wasm_throw(TAG, ptr)
}
