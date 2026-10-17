Portable SIMD module.

This module offers a portable abstraction for SIMD operations
that is not bound to any particular hardware architecture.

# What is "portable"?

This module provides a SIMD implementation that is fast and predictable on any target.

### Portable SIMD works on every target

Unlike target-specific SIMD in `std::arch`, portable SIMD compiles for every target.
In this regard, it is just like "regular" Rust.

### Portable SIMD is consistent between targets

A program using portable SIMD can expect identical behavior on any target.
In most regards, [`Simd<T, N>`] can be thought of as a parallelized `[T; N]` and operates like a sequence of `T`.

This has one notable exception: a handful of older architectures (e.g. `armv7` and `powerpc`) flush [subnormal](`f32::is_subnormal`) `f32` values to zero.
On these architectures, subnormal `f32` input values are replaced with zeros, and any operation producing subnormal `f32` values produces zeros instead.
This doesn't affect most architectures or programs.

### Operations use the best instructions available

Operations provided by this module compile to the best available SIMD instructions.

Portable SIMD is not a low-level vendor library, and operations in portable SIMD _do not_ necessarily map to a single instruction.
Instead, they map to a reasonable implementation of the operation for the target.

Consistency between targets is not compromised to use faster or fewer instructions.
In some cases, `std::arch` will provide a faster function that has slightly different behavior than the `std::simd` equivalent.
For example, `_mm_min_ps`[^1] can be slightly faster than [`SimdFloat::simd_min`](`num::SimdFloat::simd_min`), but does not conform to the IEEE standard also used by [`f32::min`].
When necessary, [`Simd<T, N>`] can be converted to the types provided by `std::arch` to make use of target-specific functions.

Many targets simply don't have SIMD, or don't support SIMD for a particular element type.
In those cases, regular scalar operations are generated instead.

[^1]: `_mm_min_ps(x, y)` is equivalent to `x.simd_lt(y).select(x, y)`
