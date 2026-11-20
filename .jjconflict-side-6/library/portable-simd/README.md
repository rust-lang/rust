# The Rust standard library's portable SIMD API
![Build Status](https://github.com/rust-lang/portable-simd/actions/workflows/ci.yml/badge.svg?branch=master)

Code repository for the [Portable SIMD Project Group](https://github.com/rust-lang/project-portable-simd).
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for our contributing guidelines.

The docs for this crate are published from the main branch.
You can [read them here][docs].

If you have questions about SIMD, we have begun writing a [guide][simd-guide].
We can also be found on [Zulip][zulip-project-portable-simd].

If you are interested in support for a specific architecture, you may want [stdarch] instead.

## Hello World

Now we're gonna dip our toes into this world with a small SIMD "Hello, World!" example. Make sure your compiler is up to date and using `nightly`. We can do that by running 

```bash
rustup update -- nightly
```

or by setting up `rustup default nightly` or else with `cargo +nightly {build,test,run}`. After updating, run 
```bash
cargo new hellosimd
```
to create a new crate. Finally write this in `src/main.rs`:
```rust
#![feature(portable_simd)]
use std::simd::f32x4;
fn main() {
    let a = f32x4::splat(10.0);
    let b = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    println!("{:?}", a + b);
}
```

Explanation: We construct our SIMD vectors with methods like `splat` or `from_array`. Next, we can use operators like `+` on them, and the appropriate SIMD instructions will be carried out. When we run `cargo run` you should get `[11.0, 12.0, 13.0, 14.0]`.

## Supported vectors

Currently, vectors may have up to 64 elements, but aliases are provided only up to 512-bit vectors.

Depending on the size of the primitive type, the number of lanes the vector will have varies. For example, 128-bit vectors have four `f32` lanes and two `f64` lanes.

The supported element types are as follows:
* **Floating Point:** `f32`, `f64`
* **Signed Integers:** `i8`, `i16`, `i32`, `i64`, `isize` (`i128` excluded)
* **Unsigned Integers:** `u8`, `u16`, `u32`, `u64`, `usize` (`u128` excluded)
* **Pointers:** `*const T` and `*mut T` (zero-sized metadata only)
* **Masks:** 8-bit, 16-bit, 32-bit, 64-bit, and `usize`-sized masks

Floating point, signed integers, unsigned integers, and pointers are the [primitive types](https://doc.rust-lang.org/core/primitive/index.html) you're already used to.
The mask types have elements that are "truthy" values, like `bool`, but have an unspecified layout because different architectures prefer different layouts for mask types.

[simd-guide]: ./beginners-guide.md
[zulip-project-portable-simd]: https://rust-lang.zulipchat.com/#narrow/stream/257879-project-portable-simd
[stdarch]: https://github.com/rust-lang/stdarch
[docs]: https://rust-lang.github.io/portable-simd/core_simd
