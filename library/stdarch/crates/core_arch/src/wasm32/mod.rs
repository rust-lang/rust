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

/// A WebAssembly `externref`: an opaque, unforgeable reference to a host
/// value, valid only while it remains live on the wasm stack.
///
/// `externref` is a bare-position-only type: it may appear only as the
/// top-level type of a function parameter, return value or local binding
/// (function pointer signature slots included). It cannot appear inside any
/// other type — no references, aggregates, statics or generic arguments —
/// which is enforced at type-check time.
///
/// The primary use is typing `extern "C"` imports and exports, where values
/// cross the host boundary directly and identity-preserving, with liveness
/// traced by the host GC:
///
/// ```ignore (wasm-only)
/// unsafe extern "C" {
///     fn create_ref() -> externref;
///     fn use_ref(v: externref);
/// }
/// ```
#[allow(non_camel_case_types)]
#[lang = "externref"]
#[non_exhaustive]
#[derive(Copy, Clone)]
#[unstable(feature = "wasm_externref", issue = "none")]
pub struct externref;

#[unstable(feature = "wasm_externref", issue = "none")]
impl !Send for externref {}

#[unstable(feature = "wasm_externref", issue = "none")]
impl !Sync for externref {}

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
    crate::intrinsics::ceilf32(a)
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
    crate::intrinsics::floorf32(a)
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
    crate::intrinsics::truncf32(a)
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
    crate::intrinsics::sqrtf32(a)
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
    crate::intrinsics::ceilf64(a)
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
    crate::intrinsics::floorf64(a)
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
    crate::intrinsics::truncf64(a)
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
    crate::intrinsics::sqrtf64(a)
}
