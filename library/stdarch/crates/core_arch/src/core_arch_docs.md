SIMD and vendor intrinsics module.

This module is intended to be the gateway to architecture-specific
intrinsic functions, typically related to SIMD (but not always!). Each
architecture that Rust compiles to may contain a submodule here, which
means that this is not a portable module! If you're writing a portable
library take care when using these APIs!

Under this module you'll find an architecture-named module, such as
`x86_64`. Each `#[cfg(target_arch)]` that Rust can compile to may have a
module entry here, only present on that particular target. For example the
`i686-pc-windows-msvc` target will have an `x86` module here, whereas
`x86_64-pc-windows-msvc` has `x86_64`.

[rfc]: https://github.com/rust-lang/rfcs/pull/2325
[tracked]: https://github.com/rust-lang/rust/issues/48556

# Overview

This module exposes vendor-specific intrinsics that typically correspond to
a single machine instruction. These intrinsics are not portable: their
availability is architecture-dependent, and not all machines of that
architecture might provide the intrinsic.

The `arch` module is intended to be a low-level implementation detail for
higher-level APIs. Using it correctly can be quite tricky as you need to
ensure at least a few guarantees are upheld:

* The correct architecture's module is used. For example the `arm` module
  isn't available on the `x86_64-unknown-linux-gnu` target. This is
  typically done by ensuring that `#[cfg]` is used appropriately when using
  this module.
* The CPU the program is currently running on supports the function being
  called. For example it is unsafe to call an AVX2 function on a CPU that
  doesn't actually support AVX2.

As a result of the latter of these guarantees all intrinsics in this module
are `unsafe` and extra care needs to be taken when calling them!

# CPU Feature Detection

In order to call these APIs in a safe fashion there's a number of
mechanisms available to ensure that the correct CPU feature is available
to call an intrinsic. Let's consider, for example, the `_mm256_add_epi64`
intrinsics on the `x86` and `x86_64` architectures. This function requires
the AVX2 feature as [documented by Intel][intel-dox] so to correctly call
this function we need to (a) guarantee we only call it on `x86`/`x86_64`
and (b) ensure that the CPU feature is available

[intel-dox]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_add_epi64&expand=100

## Static CPU Feature Detection

The first option available to us is to conditionally compile code via the
`#[cfg]` attribute. CPU features correspond to the `target_feature` cfg
available, and can be used like so:

```ignore
#[cfg(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )
)]
fn foo() {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_add_epi64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_add_epi64;

    unsafe {
        _mm256_add_epi64(...);
    }
}
```

Here we're using `#[cfg(target_feature = "avx2")]` to conditionally compile
this function into our module. This means that if the `avx2` feature is
*enabled statically* then we'll use the `_mm256_add_epi64` function at
runtime. The `unsafe` block here can be justified through the usage of
`#[cfg]` to only compile the code in situations where the safety guarantees
are upheld.

Statically enabling a feature is typically done with the `-C
target-feature` or `-C target-cpu` flags to the compiler. For example if
your local CPU supports AVX2 then you can compile the above function with:

```sh
$ RUSTFLAGS='-C target-cpu=native' cargo build
```

Or otherwise you can specifically enable just the AVX2 feature:

```sh
$ RUSTFLAGS='-C target-feature=+avx2' cargo build
```

Note that when you compile a binary with a particular feature enabled it's
important to ensure that you only run the binary on systems which satisfy
the required feature set.

## Dynamic CPU Feature Detection

Sometimes statically dispatching isn't quite what you want. Instead you
might want to build a portable binary that runs across a variety of CPUs,
but at runtime it selects the most optimized implementation available. This
allows you to build a "least common denominator" binary which has certain
sections more optimized for different CPUs.

Taking our previous example from before, we're going to compile our binary
*without* AVX2 support, but we'd like to enable it for just one function.
We can do that in a manner like:

```ignore
fn foo() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { foo_avx2() };
        }
    }

    // fallback implementation without using AVX2
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn foo_avx2() {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_add_epi64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_add_epi64;

    unsafe { _mm256_add_epi64(...); }
}
```

There's a couple of components in play here, so let's go through them in
detail!

* First up we notice the `is_x86_feature_detected!` macro. Provided by
  the standard library, this macro will perform necessary runtime detection
  to determine whether the CPU the program is running on supports the
  specified feature. In this case the macro will expand to a boolean
  expression evaluating to whether the local CPU has the AVX2 feature or
  not.

  Note that this macro, like the `arch` module, is platform-specific. For
  example calling `is_x86_feature_detected!("avx2")` on ARM will be a
  compile time error. To ensure we don't hit this error a statement level
  `#[cfg]` is used to only compile usage of the macro on `x86`/`x86_64`.

* Next up we see our AVX2-enabled function, `foo_avx2`. This function is
  decorated with the `#[target_feature]` attribute which enables a CPU
  feature for just this one function. Using a compiler flag like `-C
  target-feature=+avx2` will enable AVX2 for the entire program, but using
  an attribute will only enable it for the one function. Usage of the
  `#[target_feature]` attribute currently requires the function to also be
  `unsafe`, as we see here. This is because the function can only be
  correctly called on systems which have the AVX2 (like the intrinsics
  themselves).

And with all that we should have a working program! This program will run
across all machines and it'll use the optimized AVX2 implementation on
machines where support is detected.

# Ergonomics

It's important to note that using the `arch` module is not the easiest
thing in the world, so if you're curious to try it out you may want to
brace yourself for some wordiness!

The primary purpose of this module is to enable stable crates on crates.io
to build up much more ergonomic abstractions which end up using SIMD under
the hood. Over time these abstractions may also move into the standard
library itself, but for now this module is tasked with providing the bare
minimum necessary to use vendor intrinsics on stable Rust.

# Other architectures

This documentation is only for one particular architecture, you can find
others at:

* [`x86`]
* [`x86_64`]
* [`arm`]
* [`aarch64`]
* [`riscv32`]
* [`riscv64`]
* [`mips`]
* [`mips64`]
* [`powerpc`]
* [`powerpc64`]
* [`nvptx`]
* [`wasm32`]
* [`loongarch64`]
* [`s390x`]

[`x86`]: ../../core/arch/x86/index.html
[`x86_64`]: ../../core/arch/x86_64/index.html
[`arm`]: ../../core/arch/arm/index.html
[`aarch64`]: ../../core/arch/aarch64/index.html
[`riscv32`]: ../../core/arch/riscv32/index.html
[`riscv64`]: ../../core/arch/riscv64/index.html
[`mips`]: ../../core/arch/mips/index.html
[`mips64`]: ../../core/arch/mips64/index.html
[`powerpc`]: ../../core/arch/powerpc/index.html
[`powerpc64`]: ../../core/arch/powerpc64/index.html
[`nvptx`]: ../../core/arch/nvptx/index.html
[`wasm32`]: ../../core/arch/wasm32/index.html
[`loongarch64`]: ../../core/arch/loongarch64/index.html
[`s390x`]: ../../core/arch/s390x/index.html

# Examples

First let's take a look at not actually using any intrinsics but instead
using LLVM's auto-vectorization to produce optimized vectorized code for
AVX2 and also for the default platform.

```rust
fn main() {
    let mut dst = [0];
    add_quickly(&[1], &[2], &mut dst);
    assert_eq!(dst[0], 3);
}

fn add_quickly(a: &[u8], b: &[u8], c: &mut [u8]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_quickly_avx2(a, b, c) };
        }
    }

    add_quickly_fallback(a, b, c)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_quickly_avx2(a: &[u8], b: &[u8], c: &mut [u8]) {
    add_quickly_fallback(a, b, c) // the function below is inlined here
}

fn add_quickly_fallback(a: &[u8], b: &[u8], c: &mut [u8]) {
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = *a + *b;
    }
}
```

Next up let's take a look at an example of manually using intrinsics. Here
we'll be using SSE4.1 features to implement hex encoding.

```
fn main() {
    let mut dst = [0; 32];
    hex_encode(b"\x01\x02\x03", &mut dst);
    assert_eq!(&dst[..6], b"010203");

    let mut src = [0; 16];
    for i in 0..16 {
        src[i] = (i + 1) as u8;
    }
    hex_encode(&src, &mut dst);
    assert_eq!(&dst, b"0102030405060708090a0b0c0d0e0f10");
}

pub fn hex_encode(src: &[u8], dst: &mut [u8]) {
    let len = src.len().checked_mul(2).unwrap();
    assert!(dst.len() >= len);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { hex_encode_sse41(src, dst) };
        }
    }

    hex_encode_fallback(src, dst)
}

// translated from
// <https://github.com/Matherunner/bin2hex-sse/blob/master/base16_sse4.cpp>
#[target_feature(enable = "sse4.1")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hex_encode_sse41(mut src: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let ascii_zero = _mm_set1_epi8(b'0' as i8);
        let nines = _mm_set1_epi8(9);
        let ascii_a = _mm_set1_epi8((b'a' - 9 - 1) as i8);
        let and4bits = _mm_set1_epi8(0xf);

        let mut i = 0_isize;
        while src.len() >= 16 {
            let invec = _mm_loadu_si128(src.as_ptr() as *const _);

            let masked1 = _mm_and_si128(invec, and4bits);
            let masked2 = _mm_and_si128(_mm_srli_epi64(invec, 4), and4bits);

            // return 0xff corresponding to the elements > 9, or 0x00 otherwise
            let cmpmask1 = _mm_cmpgt_epi8(masked1, nines);
            let cmpmask2 = _mm_cmpgt_epi8(masked2, nines);

            // add '0' or the offset depending on the masks
            let masked1 = _mm_add_epi8(
                masked1,
                _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask1),
            );
            let masked2 = _mm_add_epi8(
                masked2,
                _mm_blendv_epi8(ascii_zero, ascii_a, cmpmask2),
            );

            // interleave masked1 and masked2 bytes
            let res1 = _mm_unpacklo_epi8(masked2, masked1);
            let res2 = _mm_unpackhi_epi8(masked2, masked1);

            _mm_storeu_si128(dst.as_mut_ptr().offset(i * 2) as *mut _, res1);
            _mm_storeu_si128(
                dst.as_mut_ptr().offset(i * 2 + 16) as *mut _,
                res2,
            );
            src = &src[16..];
            i += 16;
        }

        let i = i as usize;
        hex_encode_fallback(src, &mut dst[i * 2..]);
    }
}

fn hex_encode_fallback(src: &[u8], dst: &mut [u8]) {
    fn hex(byte: u8) -> u8 {
        static TABLE: &[u8] = b"0123456789abcdef";
        TABLE[byte as usize]
    }

    for (byte, slots) in src.iter().zip(dst.chunks_mut(2)) {
        slots[0] = hex((*byte >> 4) & 0xf);
        slots[1] = hex(*byte & 0xf);
    }
}
```
