# stdsimd - Rust's standard library portable SIMD API
[![Build Status](https://travis-ci.com/rust-lang/stdsimd.svg?branch=master)](https://travis-ci.com/rust-lang/stdsimd)

Code repository for the [Portable SIMD Project Group](https://github.com/rust-lang/project-portable-simd).
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for our contributing guidelines.

If you have questions about SIMD, we have begun writing a [guide][simd-guide].
We can also be found on [Zulip][zulip-project-portable-simd].

If you are interested in support for a specific architecture, you may want [stdarch] instead.

## Code Organization

Currently the crate is organized so that each element type is a file, and then the 64-bit, 128-bit, 256-bit, and 512-bit vectors using those types are contained in said file.

All types are then exported as a single, flat module.

Depending on the size of the primitive type, the number of lanes the vector will have varies. For example, 128-bit vectors have four `f32` lanes and two `f64` lanes.

The supported element types are as follows:
* **Floating Point:** `f32`, `f64`
* **Signed Integers:** `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
* **Unsigned Integers:** `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
* **Masks:** `mask8`, `mask16`, `mask32`, `mask64`, `mask128`, `masksize`

Floating point, signed integers, and unsigned integers are the [primitive types](https://doc.rust-lang.org/core/primitive/index.html) you're already used to.
The `mask` types are "truthy" values, but they use the number of bits in their name instead of just 1 bit like a normal `bool` uses.

[simd-guide]: ./beginners-guide.md
[zulip-project-portable-simd]: https://rust-lang.zulipchat.com/#narrow/stream/257879-project-portable-simd
[stdarch]: https://github.com/rust-lang/stdarch
