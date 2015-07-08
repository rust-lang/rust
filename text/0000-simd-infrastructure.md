- Feature Name: simd_basics
- Start Date: 2015-06-02
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Lay the ground work for building powerful SIMD functionality.

# Motivation

SIMD (Single-Instruction Multiple-Data) is an important part of
performant modern applications. Most CPUs used for that sort of task
provide dedicated hardware and instructions for operating on multiple
values in a single instruction, and exposing this is an important part
of being a low-level language.

This RFC lays the ground-work for building nice SIMD functionality,
but doesn't fill everything out. The goal here is to provide the raw
types and access to the raw instructions on each platform.

## Where does this code go? Aka. why not in `std`?

This RFC is focused on building stable, powerful SIMD functionality in
external crates, not `std`.

This makes it much easier to support functionality only "occasionally"
available with Rust's preexisting `cfg` system. If it were in `std`,
there would need to be some highly delayed `cfg` system so that
functions that only work with (say) AVX-2 support:

- don't break compilation on systems that don't support it, but
- are still usable on systems that do support it.

With an external crate, we can leverage `cargo`'s existing build
infrastructure: compiling with some target features will rebuild with
those features enabled.


# Detailed design

The design comes in three parts:

- types
- operations
- platform detection

The general idea is to avoid bad performance cliffs, so that an
intrinsic call in Rust maps to preferably one CPU instruction, or, if
not, the "optimal" sequence required to do the given operation
anyway. This means exposing a *lot* of platform specific details,
since platforms behave very differently: both across architecture
families (x86, x86-64, ARM, MIPS, ...), and even within a family
(x86-64's Skylake, Haswell, Nehalem, ...).

There is definitely a common core of SIMD functionality shared across
many platforms, but this RFC doesn't try to extract that, it is just
building tools that can be wrapped into a more uniform API later.


## Types & traits

There are two new attributes: `repr(simd)` and `simd_primitive_trait`

```rust
#[repr(simd)]
struct f32x4(f32, f32, f23, f23);

#[repr(simd)]
struct Simd2<T>(T, T);

#[simd_primitive_trait]
trait SimdPrim {}
```

### `repr(simd)`

The `simd` `repr` can be attached to a struct and will cause such a
struct to be compiled to a SIMD vector. It can be generic, but it is
required that any fully monomorphised instance of the type consist of
only a single "primitive" type, repeated some number of times. The
restrictions on the element type are exactly the same restrictions as
`#[simd_primitive_trait]` traits impose on their implementing types.

The `repr(simd)` may not enforce that the trait bound exists/does the
right thing at the type checking level for generic `repr(simd)`
types. As such, it will be possible to get the code-generator to error
out (ala the old `transmute` size errosr), however, this shouldn't
cause problems in practice: libraries wrapping this functionality
would layer type-safety on top (i.e. generic `repr(simd)` types would
use the `SimdPrim` trait as a bound).

It is illegal to take an internal reference to the fields of a
`repr(simd)` type, because the representation of booleans may require
to change, so that booleans are bit-packed.

### `simd_primitive_trait`

Traits marked with the `simd_primitive_trait` attribute are special:
types implementing it are those that can be stored in SIMD
vectors. Initially, only primitives and single-field structs that
store `SimdPrim` types will be allowed to implement it.

This is explicitly not a lang item: it is legal to have multiple
distinct traits in a compilation. The attribute just adds the
restriction and possibly tweaks type's internal representation (as
such, it's legal for a single type to implement multiple traits with
the attribute, if a bit pointless).

## Operations

CPU vendors usually offer "standard" C headers for their CPU specific
operations, such as [`arm_neon.h`][armneon] and [the `...mmintrin.h` headers for
x86(-64)][x86].

[armneon]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
[x86]: https://software.intel.com/sites/landingpage/IntrinsicsGuide

All of these would be exposed as (eventually) stable intrinsics with
names very similar to those that the vendor suggests (only difference
would be some form of manual namespacing, e.g. prefixing with the CPU
target), loadable via an `extern` block with an appropriate ABI.

```rust
extern "rust-intrinsic" {
    fn x86_mm_abs_epi16(a: Simd8<i16>) -> Simd8<i16>;
    // ...
}
```

These all use entirely concrete types, and this is the core interface
to these intrinsics: essentially it is just allowing code to exactly
specify a CPU instruction to use. These intrinsics only actually work
on a subset of the CPUs that Rust targets, and are only be available
for `extern`ing on those targets. The signatures are typechecked, but
in a "duck-typed" manner: it will just ensure that the types are SIMD
vectors with the appropriate length and element type, it will not
enforce a specific nominal type.

There would additionally be a small set of cross-platform operations
that are either generally efficiently supported everywhere or are
extremely useful. These won't necessarily map to a single instruction,
but will be shimmed as efficiently as possible.

- shuffles and extracting/inserting elements
- comparisons

Lastly, arithmetic and conversions are supported via built-in operators.

### Shuffles & element operations

One of the most powerful features of SIMD is the ability to rearrange
data within vectors, giving super-linear speed-ups sometimes. As such,
shuffles are exposed generally: intrinsics that represent arbitrary
shuffles.

This may violate the "one instruction per instrinsic" principal
depending on the shuffle, but rearranging SIMD vectors is extremely
useful, and providing a direct intrinsic lets the compiler (a) do the
programmers work in synthesising the optimal (short) sequence of
instructions to get a given shuffle and (b) track data through
shuffles without having to understand all the details of every
platform specific intrinsic for shuffling.

```rust
extern "rust-intrinsic" {
    fn simd_shuffle2<T: SimdVector>(v: T, w: T, i0: u32, i1: u32) -> Simd2<T::Elem>;
    fn simd_shuffle4<T: SimdVector>(v: T, w: T, i0: u32, i1: u32, i2: u32, i3: u32) -> Simd4<T::Elem>;
    fn simd_shuffle8<T: SimdVector>(v: T, w: T,
                                    i0: u32, i1: u32, i2: u32, i3: u32,
                                    i4: u32, i5: u32, i6: u32, i7: u32) -> Simd8<T::Elem>;
    fn simd_shuffle16<T: SimdVector>(v: T, w: T,
                                     i0: u32, i1: u32, i2: u32, i3: u32,
                                     i4: u32, i5: u32, i6: u32, i7: u32
                                     i8: u32, i9: u32, i10: u32, i11: u32,
                                     i12: u32, i13: u32, i14: u32, i15: u32) -> Simd16<T::Elem>;
}
```

This approach has some downsides: `simd_shuffle32` (e.g. `Simd32<u8>`
on AVX, and `Simd32<u16>` on AVX-512) and especially `simd_shuffle64`
(e.g. `Simd64<u8>` on AVX-512) are unwieldy. These have similar type
"safety"/code-generation errors to the vectors themselves.

These operations are semantically:

```rust
// vector of double length
let z = concat(v, w);

return [z[i0], z[i1], z[i2], ...]
```

The indices `iN` have to be compile time constants.

Similarly, intrinsics for inserting/extracting elements into/out of
vectors are provided, to allow modelling the SIMD vectors as actual
CPU registers as much as possible:

```rust
extern "rust-intrinsic" {
    fn simd_insert<T: SimdVector>(v: T, i0: u32, elem: T::Elem) -> T;
    fn simd_extract<T: SimdVector>(v: T, i0: u32) -> T::Elem;
}
```

The `i0` indices do not have to be constant. These are equivalent to
`v[i0] = elem` and `v[i0]` respectively.

### Comparisons

Comparisons are implemented via intrinsics, because the current
comparison operator infrastructure doesn't easily lend itself to
return vectors, as required.

The raw signatures would look like:

```rust
extern "rust-intrinsic" {
    fn simd_eq<T, U>(v: T, w: T) -> U;
    fn simd_ne<T, U>(v: T, w: T) -> U;
    fn simd_lt<T, U>(v: T, w: T) -> U;
    fn simd_le<T, U>(v: T, w: T) -> U;
    fn simd_gt<T, U>(v: T, w: T) -> U;
    fn simd_ge<T, U>(v: T, w: T) -> U;
}
```

However, these will be type checked, to ensure that `T` and `U` are
the same length, and that `U` is appropriately shaped for a boolean. A
library actually importing them might use some trait bounds to get
actual type-safety.

### Built-in functionality

Any type marked `repr(simd)` automatically has the `+`, `-` and `*`
operators work. The `/` operator works for floating point, and the
`<<` and `>>` ones work for integers.

SIMD vectors can be converted with `as`. As with intrinsics, this is
"duck-typed" it is possible to cast a vector type `V` to a type `W` if
their lengths match and their elements are castable (i.e. are
primitives), there's no enforcement of nominal types.

All of these are never checked: explicit SIMD is essentially only
required for speed, and checking inflates one instruction to 5 or
more.

## Platform Detection

The availability of efficient SIMD functionality is very fine-grained,
and our current `cfg(target_arch = "...")` is not precise enough. This
RFC proposes a `target_feature` `cfg`, that would be set to the
features of the architecture that are known to be supported by the
exact target e.g.

- a default x86-64 compilation would essentially only set
  `target_feature = "sse"` and `target_feature = "sse2"`
- compiling with `-C target-feature="+sse4.2"` would set
  `target_feature = "sse4.2"`, `target_feature = "sse.4.1"`, ...,
  `target_feature = "sse"`.
- compiling with `-C target-cpu=native` on a modern CPU might set
  `target_feature = "avx2"`, `target_feature = "avx"`, ...

The possible values of `target_feature` will be a selected whitelist,
not necessarily just everything LLVM understands. There are other
non-SIMD features that might have `target_feature`s set too, such as
`popcnt` and `rdrnd` on x86/x86-64.)

With a `cfg_if_else!` macro that expands to the first `cfg` that is
satisfied (ala [@alexcrichton's cascade][cascade]), code might look
like:

[cascade]: https://github.com/alexcrichton/backtrace-rs/blob/03703031babfa87cbe2c723ad6752131819dc554/src/macros.rs

```rust
cfg_if_else! {
    if #[cfg(target_feature = "avx")] {
        fn foo() { /* use AVX things */ }
    } else if #[cfg(target_feature = "sse4.1")] {
        fn foo() { /* use SSE4.1 things */ }
    } else if #[cfg(target_feature = "sse2")] {
        fn foo() { /* use SSE2 things */ }
    } else if #[cfg(target_feature = "neon")] {
        fn foo() { /* use NEON things */ }
    } else {
        fn foo() { /* universal fallback */ }
    }
}
```

# Extensions

- scatter/gather operations allow (partially) operating on a SIMD
  vector of pointers. This would require extending `SimdPrim` to also
  allow pointer types.
- allow (and ignore for everything but type checking) zero-sized types
  in `repr(simd)` structs, to allow tagging them with markers

# Alternatives

- The SIMD on-route-to-stable intrinsics could have their own ABI
- Intrinsics could instead by namespaced by ABI, `extern
  "x86-intrinsic"`, `extern "arm-intrinsic"`.
- There could be more syntactic support for shuffles, either with true
  syntax, or with a syntax extension. The latter might look like:
  `shuffle![x, y, i0, i1, i2, i3, i4, ...]`. However, this requires
  that shuffles are restricted to a single type only (i.e. `Simd4<T>`
  can be shuffled to `Simd4<T>` but nothing else), or some sort of
  type synthesis. The compiler has to somehow work out the return
  value:

  ```rust
  let x: Simd4<u32> = ...;
  let y: Simd4<u32> = ...;

  // reverse all the elements.
  let z = shuffle![x, y, 7, 6, 5, 4, 3, 2, 1, 0];
  ```

  Presumably `z` should be `Simd8<u32>`, but it's not obvious how the
  compiler can know this. The `repr(simd)` approach means there may be
  more than one SIMD-vector type with the `Simd8<u32>` shape (or, in
  fact, there may be zero).
- Instead of platform detection, there could be feature detection
  (e.g. "platform supports something equivalent to x86's `DPPS`"), but
  there probably aren't enough cross-platform commonalities for this
  to be worth it. (Each "feature" would essentially be a platform
  specific `cfg` anyway.)
- Check vector operators in debug mode just like the scalar versions.

# Unresolved questions

- Should integer vectors get `/` and `%` automatically? Most CPUs
  don't support them for vectors.
