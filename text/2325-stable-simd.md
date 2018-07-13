- Feature Name: `vendor_intrinsics`
- Start Date: 2018-02-04
- RFC PR: [rust-lang/rfcs#2325](https://github.com/rust-lang/rfcs/pull/2325)
- Rust Issue: [rust-lang/rust#48556](https://github.com/rust-lang/rust/issues/48556)

# Summary
[summary]: #summary

The purpose of this RFC is to provide a framework for SIMD to be used on stable
Rust. It proposes stabilizing x86-specific vendor intrinsics, but includes the
scaffolding for other platforms as well as a future portable SIMD design (to be
fleshed out in another RFC).

# Motivation
[motivation]: #motivation

Stable Rust today does not typically expose all of the capabilities of the
platform that you're running on. A notable gap in Rust's support includes SIMD
(single instruction multiple data) support. For example on x86 you don't
currently have explicit access to the 128, 256, and 512 bit registers on the
CPU. LLVM is in general an excellent optimizing compiler and often attempts to
make use of these registers (auto vectorizing code), but it unfortunately is
still somewhat limited and doesn't express the full power of the various SIMD
intrinsics.

The goal of this RFC is to enable using SIMD intrinsics on stable Rust, and in
general provide a means to access the architecture-specific functionality of
each vendor. For example the AES intrinsics on x86 would also be made available
through this RFC, not only the SIMD-related AVX intrinsics.

Note that this is certainly not the first discussion to broach the topic of SIMD
in Rust, but rather this has been an ongoing discussion for quite some time now!
For example the [simd crate][simd-crate] started [long ago][simd-start], we've
had [rfcs][simd-rfc], we've had a [lot][i1] of [discussions][i2] on internals,
and the [stdsimd] crate has been implemented.

This RFC draws from much of the historical feedback and design that we've done
around SIMD in Rust and is targeted at providing path forward for using SIMD on
stable Rust while allowing the compiler to change in the future and retain a
stable interface.

[simd-rfc]: https://github.com/rust-lang/rfcs/pull/1199
[simd-crate]: https://github.com/rust-lang-nursery/simd
[simd-start]: http://huonw.github.io/blog/2015/08/simd-in-rust/
[stdsimd]: https://github.com/rust-lang-nursery/stdsimd
[i1]: https://internals.rust-lang.org/t/getting-explicit-simd-on-stable-rust/4380
[i2]: https://internals.rust-lang.org/t/whats-the-next-step-towards-the-stabilization-of-simd/5867

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Let's say you've just heard about this fancy feature called "auto vectorization"
in LLVM and you want to take advantage of it. For example you've got a function
like this you'd like to make faster:

```rust
pub fn foo(a: &[u8], b: &[u8], c: &mut [u8]) {
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = *a + *b;
    }
}
```

When [inspecting the assembly][asm1] you notice that rustc is making use of the
`%xmmN` registers which you've read is related to SSE on your CPU. You know,
however, that your CPU supports up to AVX2 which has bigger registers, so you'd
like to get access to them!

Your first solution to this problem is to compile with `-C
target-feature=+avx2`, and after that you see the `%ymmN` registers being used,
yay! Unfortunately though you're publishing this binary on CPUs which may not
actually have AVX2 as a feature, so you don't want to enable AVX2 for the entire
program. Instead what you can do is enable it for just this function:

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn foo(a: &[u8], b: &[u8], c: &mut [u8]) {
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = *a + *b;
    }
}
```

And [sure enough][asm2] you see the `%ymmN` registers getting used in this
function! Note, however, that because you've explicitly enabled a feature you're
required to declare the function as `unsafe`, as specified in [RFC
2045][rfc2045] (although this requirement is likely to be relaxed in [RFC
2212][rfc2212]). This worked as a proof of concept but what you still need to do
is dispatch at runtime whether the local CPU that you're running on supports
AVX2 or not. Thankfully, though, libstd has a handy macro for this!

[rfc2212]: https://github.com/rust-lang/rfcs/pull/2212

```rust
pub fn foo(a: &[u8], b: &[u8], c: &mut [u8]) {
    // Note that this `unsafe` block is safe because we're testing
    // that the `avx2` feature is indeed available on our CPU.
    if is_target_feature_detected!("avx2") {
        unsafe { foo_avx2(a, b, c) }
    } else {
        foo_fallback(a, b, c)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn foo_avx2(a: &[u8], b: &[u8], c: &mut [u8]) {
    foo_fallback(a, b, c) // the function below is inlined here
}

fn foo_fallback(a: &[u8], b: &[u8], c: &mut [u8]) {
    for ((a, b), c) in a.iter().zip(b).zip(c) {
        *c = *a + *b;
    }
}
```

And [sure enough once again][asm3] we see that `foo` is dispatching at runtime
to the appropriate function, and only `foo_avx2` is using our `%ymmN` registers!

[asm1]: https://play.rust-lang.org/?gist=36b253cd70840ea2ce6aad90418ec58b&version=nightly&mode=release
[asm2]: https://play.rust-lang.org/?gist=a31bdd3ce2b9a60e3317ccafb0133490&version=nightly&mode=release
[asm3]: https://play.rust-lang.org/?gist=cfce0743910291517aae3b15f70a7cbd&version=nightly&mode=release
[rfc2045]: https://github.com/rust-lang/rfcs/blob/master/text/2045-target-feature.md

Ok great! At this point we've seen how to enable CPU features for
functions-at-a-time as well as how they could be used in a larger context to do
runtime dispatch to the most appropriate implementation. As we saw in the
motivation, however, we're just relying on LLVM to auto-vectorize here which
often isn't good enough or otherwise doesn't expose the functionality we want.

For **explicit and guaranteed simd** on stable Rust you'll be using a new module
in the standard library, `std::arch`. The `std::arch` module is defined by
vendors/architectures, not us actually! For example Intel [publishes a list of
intrinsics][intel-intr] as does [ARM][arm-intr]. These exact functions and their
signatures will be available in `std::arch` with types translated to Rust
(e.g. `int32_t` becomes `i32`). Vendor specific types like `__m128i` on Intel
will also live in `std::arch`.

For example let's say that we're writing a function that encodes a `&[u8]` in
ascii hex and we want to convert `&[1, 2]` to `"0102"`. The [stdsimd]
crate currently has this [as an example][hex-example], and let's take a look at
a few snippets from that.

First up you'll see the dispatch routine like we wrote above:

```rust
fn hex_encode<'a>(src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, usize> {
    let len = src.len().checked_mul(2).unwrap();
    if dst.len() < len {
        return Err(len);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_target_feature_detected!("avx2") {
            return unsafe { hex_encode_avx2(src, dst) };
        }
        if is_target_feature_detected!("sse4.1") {
            return unsafe { hex_encode_sse41(src, dst) };
        }
    }

    hex_encode_fallback(src, dst)
}
```

Here we have some routine business about hex encoding in general, and then for
x86/x86\_64 platforms we have optimized versions specifically for avx2 and
sse41. Using the `is_target_feature_detected!` macro in libstd we saw above
we'll dispatch to the correct one at runtime.

Taking a closer look at [`hex_encode_sse41`] we see that it starts out with a
bunch of weird looking function calls:

```rust
let ascii_zero = _mm_set1_epi8(b'0' as i8);
let nines = _mm_set1_epi8(9);
let ascii_a = _mm_set1_epi8((b'a' - 9 - 1) as i8);
let and4bits = _mm_set1_epi8(0xf);
```

As it turns out though, these are all Intel SIMD intrinsics! For example
[`_mm_set1_epi8`] is defined as creating an instance of `__m128i`, a 128-bit
integer register. The intrinsic specificall sets all bytes to the first
argument.

These functions are all imported through `std::arch::*` at the top of the
example (in this case `stdsimd::vendor::*`). We go on to use a bunch of these
intrinsics throughout the `hex_encode_sse41` function to actually do the hex
encoding.

The example listed currently has some tests/benchmarks as well, and if we run
the benchmarks we'll see:

```
test benches::large_default    ... bench:      73,432 ns/iter (+/- 12,526) = 14279 MB/s
test benches::large_fallback   ... bench:   1,711,030 ns/iter (+/- 286,642) = 612 MB/s
test benches::small_default    ... bench:          30 ns/iter (+/- 18) = 3900 MB/s
test benches::small_fallback   ... bench:         204 ns/iter (+/- 74) = 573 MB/s
test benches::x86::large_avx2  ... bench:      69,742 ns/iter (+/- 9,157) = 15035 MB/s
test benches::x86::large_sse41 ... bench:     108,463 ns/iter (+/- 70,250) = 9667 MB/s
test benches::x86::small_avx2  ... bench:          25 ns/iter (+/- 8) = 4680 MB/s
test benches::x86::small_sse41 ... bench:          25 ns/iter (+/- 14) = 4680 MB/s
```

Or in other words, our runtime dispatch implementation ("default") is **20 times
faster** than the fallback implementation (no explicit SIMD). Furthermore the
AVX2 implementation is nearly 2x faster than the SSE4.1 implementation for large
inputs, and the SSE4.1 implementation is over 10x faster than the default
fallback as well.

With `std::arch` and `is_target_feature_detected!` we've now written a program
that's 20x faster on supported hardware, yet it also continues to run on older
hardware as well! Not bad for a few dozen lines on each function!

[intel-intr]: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
[arm-intr]: https://developer.arm.com/technologies/neon/intrinsics
[hex-example]: https://github.com/rust-lang-nursery/stdsimd/blob/ee046e0419e4d5e8f742b138313eeefd603326b5/examples/hex.rs
[`hex_encode_sse41`]: https://github.com/rust-lang-nursery/stdsimd/blob/ee046e0419e4d5e8f742b138313eeefd603326b5/examples/hex.rs#L114-L160
[`_mm_set1_epi8`]: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set1_epi8&expand=4669

---

Note that this RFC is explicitly not attempting to stabilize/design a set of
"portable simd operations". The contents of `std::arch` are platform specific
and provide no guarantees about portability. Efforts in the past, however, such
as with [simd.js] and the [simd crate][simd-crate] show that it's desirable and
useful to have a set of types which are usable across platforms.

Furthermore LLVM does quite a good job with a portable `u32x4` type, for
example, in terms of platform support and speed on platforms that support it.
This RFC is not going to go too much into the details about these types, but
rather these guidelines will still hold:

* The intrinsics **will not** take portable types as arguments. For example
  `u32x4` and `__m128i` will be different types on x86. The two types, however,
  will be convertible between one another (either via transmutes or via explicit
  functions). This conversion will have zero run-time cost.
* The portable simd types will likely live in a module like `std::simd` rather
  than `std::arch`.

The design around these portable types are ongoing, however, and stay tuned for
an RFC for the `std::simd` module!

[simd.js]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SIMD

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Stable SIMD in Rust ends up requiring a surprising number of both language and
library features to be productive. Thankfully, though, there's has been quite a
bit of experimentation over time with SIMD in Rust and we've gotten a lot of
good experience along the way! In this section, though, we'll be going into the
various features in detail.

## The `#[target_feature]` Attribute

The `#[target_feature]` attribute was specified in [RFC 2045][rfc2045] and
remains unchanged from that specification. As a quick recap it allows you to add
this attribute to functions:

```rust
#[target_feature(enable = "avx2")]
```

The only currently allowed key is `enable` (one day we may allow `disable`). The
string values accepted by `enable` will be separately stabilized but are likely
to be guided by vendor definitions. For example in Intel's [intrinsic
guide][intel-intr] it lists functions under "AVX2", so we're likely to stabilize
the name `avx2` for Rust.

There's a good number of these features supported by the compiler today. It's
expected that when stabilizing other pieces of this RFC the names of the
following existing features for x86 will be stabilized:

* `aes`
* `avx2`
* `avx`
* `bmi2`
* `bmi` - to be renamed to `bmi1`, the name Intel gives it
* `fma`
* `fxsr`
* `lzcnt`
* `popcnt`
* `rdrnd`
* `rdseed`
* `sse2`
* `sse3`
* `sse4.1`
* `sse4.2`
* `sse`
* `ssse3`
* `xsave`
* `xsavec`
* `xsaveopt`
* `xsaves`

Note that AVX-512 names are missing from this list, but that's because we
haven't implemented any AVX-512 intrinsics yet. Those'll get stabilized on their
own once implemented. Additionally note that `mmx` is missing from this list.
For reasons discussed later, it's proposed that MMX types are omitted from the
first pass of stabilization. AMD also has some specific features supported
(`sse4a`, `tbm`), and so do ARM, MIPS, and PowerPC, but none of these feature
names a proposed for becoming stable in the first pass.

## The `target_feature` value in `#[cfg]`

In addition to enabling target features for a function the compiler will also
allow statically testing whether a particular target feature is enabled. This
corresponds to the `cfg_target_feature` feature today in rustc, and can be seen
via:

```rust
#[cfg(target_feature = "avx")]
fn foo() {
    // implementation that can use `avx`
}

#[cfg(not(target_feature = "avx"))]
fn foo() {
    // a fallback implementation
}
```

Additionally this is also made available to `cfg!`:

```rust
if cfg!(target_feature = "avx") {
    println!("this program was compiled with AVX support");
}
```

The `#[cfg]` attribute and `cfg!` macro statically resolve and **do not do
runtime dispatch**. Tweaking these functions is currently done via the `-C
target-feature` flag to the compiler. This flag to the compiler accepts a
similar set of strings to the ones specified above and is already "stable".

## The `is_target_feature_detected!` Macro

One mode of operation with intrinsics is to compile *part* of a program with
certain CPU features enabled but not the entire program. This way a portable
program can be compiled which runs across a broad range of hardware which can
still benefit from optimized implementations for particular hardware at
runtime.

The crux of this support in libstd is this macro provided by libstd,
`is_target_feature_detected!`. The macro will accept one argument, a string
literal.  The string can be any feature passed to `#[target_feature(enable =
...)]` for the platform you're compiling for. Finally, the macro will resolve to
a `bool` result.

For example on x86 you could write:

```rust
if is_target_feature_detected!("sse4.1") {
    println!("this cpu has sse4.1 features enabled!");
}
```

It would, however, be an error to write this on x86 cpus:

```rust
is_target_feature_detected!("neon"); //~ COMPILE ERROR: neon is an ARM feature, not x86
is_target_feature_detected!("foo"); //~ COMPILE ERROR: unknown target feature for x86
```

The macro is intended to be implemented in the `std` crate (**not** `core`) and
made available via the normal macro preludes. The implementation of this macro
is expected to be what [`stdsimd`][stdsimd] does today, notably:

* The first time the macro is invoked all the local CPU features will be
  detected.
* The detected features will then be cached globally (when possible and
  currently in a bitset) for the rest of the execution of the program.
* Further invocations of `is_target_feature_detected!` are expected to be cheap
  runtime dispatches. (aka load a value and check whether a bit is set)
* Exception: in some cases the result of the macro is statically known: for
  example, `is_target_feature_detected!("sse2")` when the binary is being
  compiled with "sse42" globally. In these cases, none of the steps above are
  performed and the macro just expands to `true`.

The exact method of CPU feature detection various by platform, OS, and
architecture. For example on x86 we make heavy use of the `cpuid` instruction
whereas on ARM the implementation currently uses getauxval/`/proc` mounted
information on Linux. It's expected that the detection will vary for each
particular target, as necessary.

Note that the implementation details of the macro today prevent it from being
located in libcore. If getauxval or `/proc` is used that requires libc to be
available or `File` in one form or another. These concerns are currently
std-only (not available in libcore). This is also a conservative route for x86
where it is possible to do CPU feature detection in libcore (as it's just the
`cpuid` instruction), but for consistency across platforms the macro will only
be available in libstd for now. This placement can of course be relaxed in the
future if necessary.

## The `std::arch` Module

This is where the real meat is. A new module will be added to the standard
library, `std::arch`. This module will also be available in `core::arch`
(and `std` will simply reexport it). The contents of this module provide no
portability guarantees (like `std::os` and unlike the rest of `std`). APIs
present on one platform may not be present on another.

The contents of the `arch` modules are defined by, well, architectures! For
example Intel has an [intrinsics guide][intel-intr] which will serve as a
guideline for all contents in the `arch` module itself. The standard library
will not deviate in naming or type signature of any intrinsic defined by an
architecture.

For example most Intel intrinsics start with `_mm_` or `_mm256_` for 128 and
256-bit registers. While perhaps unergonomic, we'll be sticking to what Intel
says. Note that all intrinsics will also be `unsafe`, according to [RFC
2045][rfc2045].

Function signatures defined by architectures are typically defined in terms of C
types. In Rust, however, those aren't always available! Instead the intrinsics
will be defined in terms of Rust-specific types. Some types are easily
translated such as `int32_t`, but otherwise a different mapping may be applied
per-architecture.

The current proposed mapping for x86 intrinsics is:

| What Intel says | Rust Type |
|-----------------|-----------|
| `void*`         | `*mut u8` |
| `char`          | `i8`      |
| `short`         | `i16`     |
| `int`           | `i32`     |
| `long long`     | `i64`     |
| `const int`     | `i32` [0] |

[0] required to be compile-time constants.

Other than these exceptions the x86 intrinsics will be defined exactly as Intel
defines them. This will necessitate new types in the `std::arch` modules for
SIMD registers! For example these new types will all be present in `std::arch`
on x86 platforms:

* `__m128`
* `__m128d`
* `__m128i`
* `__m256`
* `__m256d`
* `__m256i`

(note that AVX-512 types will come in the future!)

Infrastructure-wise the contents of `std::arch` are expected to continue to be
defined in the [`stdsimd` crate/repository][stdsimd]. Intrinsics defined here go
through a rigorous test suite involving automatic verification against the
upstream architecture defintion, verification that the correct instruction is
generated by LLVM, and at least one runtime test for each intrinsic to ensure it
not only compiles but also produces correct results. It's expected that
stabilized intrinsics will meet these critera to the best of their ability.

Currently today on x86 and ARM platforms the stdsimd crate performs all these
checks, but these checks are not yet implemented for PowerPC/MIPS/etc, but
that's always just some more work to do!

It's not expected that the contents of `std::arch` will remain static for all
time. Rather intrinsics will continue to be implemented in `stdsimd` and make
their way into the main Rust repository. For example there are not currently any
implemented AVX-512 intrinsics, but that doesn't mean we won't implement them!
Rather once implemented they'll be stabilized and included in libstd following
the Rust release model.

## The types in `std::arch`

It's worth paying close attention to the types in `std::arch`. Types like
`__m128i` are intended to represent a 128-bit packed SIMD register on x86, but
there's nothing stopping you from using types like `Option<__m128i>` in your
program!  Most generic containers and such probably aren't written with packed
SIMD types in mind, and it'd be a bummer if everything stopped working once you
used a packed SIMD type in one of them.

Instead it will be required that the types defined in `std::arch` do indeed
work when used in "nonstandard" contexts. For example `Option<__m128i>` should
never produce a compiler error or a codegen error when used (it may just be
slower than you expect). This requires special care to be taken both in
representation of these arguments as well as their ABI.

Implementation-wise these packed SIMD types are implemented in terms of a
"portable" vector in LLVM. LLVM as a results gets most of this logic correct for
us in terms of having these compile without errors in many contexts. The ABI,
however, had to be special cased as it was a location where LLVM didn't always
help us.

The Rust ABI will currently be implemented such that all related packed-SIMD
types are passed via *memory* instead of by-value. This means that regardless of
the target features enabled for a function everything should agree on how packed
SIMD arguments are passed across boundaries and whatnot.

Again though, note that this section is largely an implementation detail of SIMD
in Rust today, though it's enabling usage without a lot of codegen errors
popping up all over the place.

## Intrinsics in `std::arch` and constant arguments

There are a number of intrinsics on x86 (and other) platforms that require their
arguments to be constants rather than decided at runtime. For example
[`_mm_insert_pi16`][_mm_insert_pi16] requires its third argument to be a
constant value where only the lowest two bits are used. The Rust type system,
however, does not currently have a stable way of expressing this information.

Eventually we will likely have some form of `const` arguments or `const`
machinery to guarantee that these functions are called and monomorphized with
constant arguments, but for now this RFC proposes taking a more conservative
route forward. Instead we'll, for the time being, forbid the functions from
being invoked with non-constant arguments. Prototyped in [#48018][const-pr] the
`stdsimd` crate will have an unstable attribute where the compiler can help
provide this guarantee. As an extra precaution as well [#48078][const-pr2] also
implements disallowing taking a function pointer to these intrinsics, requiring
a direct invocation.

[const-pr]: https://github.com/rust-lang/rust/pull/48018
[const-pr2]: https://github.com/rust-lang/rust/pull/48078
[_mm_insert_pi16]: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=insert_pi&expand=2973

It's hoped that this restriction will allow `stdsimd` to be forward compatible
with a future const-powered world of Rust but in the meantime not otherwise
block stabilization of these intrinsics.

## Portable packed SIMD

So-called "portable" packed SIMD types are currently implemented in both the
[stdsimd] and [simd][simd-crate] crates. These types look like `u8x16` and
explicitly specify how many lanes they have (16 in this case) and what type each
line is (`u8` in this case). These types are intended to unconditionally
available (like the rest of libstd) and simply optimized much more aggressively
on platforms that have native support for the various operations.

For example `u8x16::add` may be implemented differently on i586 vs i686, and
also entirely differently implemented on ARM. The idea with portable packed SIMD
types is that they represent a broad intersection of fast behavior across a
broad range of platforms.

It's intended that this RFC neither includes nor rules out the addition of
portable packed-SIMD types in Rust. It's expected that an upcoming RFC will
propose the addition of these types in a `std::simd` module. These types will be
orthogonal to scalable-vector types which are expected to be proposed in
another, also different, RFC.  What this RFC does do, however, is explicitly
specify that:

* The portable SIMD types (both packed and scalable) will not be used in
  intrinsics.
* The per-architecture SIMD types will be distinct types from the portable SIMD
  types.

Or, in other words, it's intended that portable SIMD types are entirely
decoupled from intrinsics. If they both end up being implemented then
there will be jkro-cost interoperation between them, but neither
will necessarily depend on the other.

[soundbug]: https://github.com/rust-lang/rust/issues/44367

## Not stabilizing MMX in this RFC

This RFC proposed notably omitting the MMX intrinsics, or those related to
`__m64` in other words. The MMX type `__m64` and the intrinsics have been
somewhat problematic in a number of ways. Known cases include:

* [MMX intrinsics aren't always desirable][mmx1]
* [LLVM codegen errors happen with debuginfo enabled and MMX][mmx2]
* [LLVM codegen errors with MMX types and i586][mmx3]

[mmx1]: https://github.com/rust-lang/rust/pull/45367#issuecomment-337883136
[mmx2]: https://github.com/rust-lang-nursery/stdsimd/issues/246
[mmx3]: https://github.com/rust-lang-nursery/stdsimd/issues/300

Due to these issues having an unclear conclusion as well as a seeming lack of
desire to stabilize MMX intrinsics, the `__m64` and all related intrinsics
**will not be stabilized** via this RFC.

# Drawbacks

[drawbacks]: #drawbacks

This RFC represents a *significant* addition to the standard library, maybe one
of the largest we've ever done! As a result alternate implementations of Rust
will likely have a difficult time catching up to rustc/LLVM with all the SIMD
intrinsics. Additionaly the semantics of "packed SIMD types should work
everywhere" may be overly difficult to implement in alternate implementations.
It is worth noting that both [Cranelift][cranelift] and GCC support packed SIMD
types.

[cranelift]: https://github.com/CraneStation/cranelift/

Due to the enormity of what's being added to the standard library it's also
infeasible to carefully review each addition in isolation. While there are a
number of automatic verifications in place we're likely to inevitably make a
mistake when stabilizing something. *Fixing* a stabilization can often be quite
difficult and costly.

# Rationale and alternatives
[alternatives]: #alternatives

Over the years quite a few iterations have happened for SIMD in Rust. This RFC
draws from as many of those as it can and attempts to strike a balance between
exposing functionality while still allowing us to implement everything in a
stable fashion for years to come (and without blocking us from updating LLVM,
for example). Despite this there's a few alternatives we could do as well.

## Portable types in architecture interfaces

It was initially attempted in the [stdsimd] crate that we would use the portable
types on all of the intrinsics. For example instead of:

```rust
pub unsafe fn _mm_set1_epi8(val: i8) -> __m128i;
```

we would instead define

```rust
pub unsafe fn _mm_set1_epi8(val: i8) -> i8x16;
```

The latter definition here is much easier for a beginner to SIMD to read (or at
least I gawked when I first saw `__m128i`).

The downside of this approach, however, is that Intel isn't telling us what to
do. While that may sound simple, this RFC is proposing an addition of
**thousands** of functions to the standard library in a stable manner. It's
infeasible for any one person (or even the entire libs team) to scrutinize all
functions and assess whether the correct signature is applied (aka was it
`i8x16` or `i16x8`?)

Furthermore not all intrinsics from Intel actually have an interpretation with
one of the portable types. For example some intrinsics take an integer constant
which when 0 interprets the input as `u8x16` and when 1 interprets it as
`u16x8` (as an example). This effectively means that there *isn't* a correct
choice in all situations for what portable type should be used.

Consequently it's proposed that instead of portable types the exact architecture
types are used in all intrinsics. This provides us a much easier route to
stabilization ("make sure it's what Intel says") along with no need to interpret
what Intel does and attempt to find the most appropriate type.

There is interest by both current `stdsimd` maintainers and users to expose
a "better-typed" SIMD API in crates.io that builds on top of the intrinsics
proposed for stabilization here.

## Stabilizing SIMD implementation details

Another alternative to the bulk of this RFC is allowing more raw access to the
internals of LLVM. For example stabilizing `#[repr(simd)]` or the ability to
write `extern "platform-intrinsics" { ... }` or `#[link_llvm_intrinsic...]`.
This is certainly a *much* smaller surface area to stabilize (aka not
thousands of intrinsics).

This avenue was decided against, however, for a few reasons:

* Such raw interfaces may change over time as they simply represent LLVM as a
  current point in time rather than what LLVM wants to do in the future.
* Alternate implementations of rustc or alternate rustc backends like
  [Cranelift][cranelift] may not expose the same sort of functionality that
  LLVM provides, or implementing the interfaces may be much more difficult in
  alternate backends than in LLVM's.

[cranelift]: https://github.com/CraneStation/cranelift/

As a result, it's intended that instead of exposing raw building blocks (and
allowing `stdsimd` to live on crates.io) we'll instead pull in `stdsimd` to the
standard library and expose it as the stable interface to SIMD in Rust.

# Unresolved questions
[unresolved]: #unresolved-questions

There's a number of unresolved questions around stabilizing SIMD today which
don't pose serious blockers and may also wish to be considered open bugs rather
than blocking stabilization:

## Relying on unexported LLVM APIs

The static detection performed by `cfg!` and `#[cfg]` currently relies on a
[Rust-specific patch to LLVM][llvm-patch]. LLVM internal knows all about
hierarchies of features and such. For example if you compile with `-C
target-feature=+avx2` then `cfg!(target_feature = "sse2")` also needs to resolve
to `true`. Rustc, however, does not know about these features and relies on
learning this information through LLVM.

Unfortunately though LLVM does not actually export this information for us to
consume (as far as we know). As a result we have a [local patch][llvm-patch]
which exposed this information for us to read. The consequence of this
implementation detail is that when compiled against the system LLVM the `cfg!`
macro may not work correctly when used in conjunction with `-C target-feature`
or `-C target-cpu` flags.

It appears that clang [vendors and/or duplicates][clang] LLVM's functionality in
this regard. It's an option for rustc to do the same but it may also be an
option to expose the information in upstream LLVM. So far there appears to have
been no attempts to upstream this patch into LLVM itself.

[llvm-patch]: https://github.com/rust-lang/llvm/commit/68e1e29618b2bd094d82faac16cf8e89959bbd68
[clang]: https://github.com/llvm-mirror/clang/blob/679d846fcc73bd213347785185006d591698a132/lib/Basic/Targets/X86.cpp

## Packed SIMD types in `extern` functions are not sound

The packed SIMD types have particular care paid to them with respect to their
ABI in Rust and how they're passed between functions, notably to ensure that
they work properly throughout Rust programs. The "fix" to pass them in memory
over function calls, however, was only applied to the "Rust" ABI and not any
other function ABIs.

A consequence of this change is that if you instead label all your functions
`extern` then the [same bug][soundbug] will arise. It may be possible to
implement a "lint" or a compiler error of sorts to forbid this situation in the
short term. We could also possibly accept this as a known bug for the time
being.

## What if we're wrong?

Despite the CI infrastructure of the `stdsimd` crate it seems inevitable that
we'll get an intrinsic wrong at some point. What do we do in a situation like
that? This situation is somewhat analagous to the `libc` crate but there you can
fix the problem downstream (just have a corrected type/definition) for
vendor intrinsics it's not so easy.

Currently it seems that our only recourse would be to add a `2` suffix to the
function name or otherwise indicate there's a corrected version, but that's not
always the best...
