- Feature Name: repr_simd, platform_intrinsics, cfg_target_feature
- Start Date: 2015-06-02
- RFC PR: https://github.com/rust-lang/rfcs/pull/1199
- Rust Issue: https://github.com/rust-lang/rust/issues/27731

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

(An earlier variant of this RFC was discussed as a
[pre-RFC](https://internals.rust-lang.org/t/pre-rfc-simd-groundwork/2343).)

## Where does this code go? Aka. why not in `std`?

This RFC is focused on building stable, powerful SIMD functionality in
external crates, not `std`.

This makes it much easier to support functionality only "occasionally"
available with Rust's preexisting `cfg` system. There's no way for
`std` to conditionally provide an API based on the target features
used for the final artifact. Building `std` in every configuration is
certainly untenable. Hence, if it were to be in `std`, there would
need to be some highly delayed `cfg` system to support that sort of
conditional API exposure.

With an external crate, we can leverage `cargo`'s existing build
infrastructure: compiling with some target features will rebuild with
those features enabled.


# Detailed design

The design comes in three parts, all on the path to stabilisation:

- types (`feature(repr_simd)`)
- operations (`feature(platform_intrinsics)`)
- platform detection (`feature(cfg_target_feature)`)

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


## Types

There is a new attribute: `repr(simd)`.

```rust
#[repr(simd)]
struct f32x4(f32, f32, f32, f32);

#[repr(simd)]
struct Simd2<T>(T, T);
```

The `simd` `repr` can be attached to a struct and will cause such a
struct to be compiled to a SIMD vector. It can be generic, but it is
required that any fully monomorphised instance of the type consist of
only a single "primitive" type, repeated some number of times.

The `repr(simd)` may not enforce that any trait bounds exists/does the
right thing at the type checking level for generic `repr(simd)`
types. As such, it will be possible to get the code-generator to error
out (ala the old `transmute` size errors), however, this shouldn't
cause problems in practice: libraries wrapping this functionality
would layer type-safety on top (i.e. generic `repr(simd)` types would
use some `unsafe` trait as a bound that is designed to only be
implemented by types that will work).

Adding `repr(simd)` to a type may increase its minimum/preferred
alignment, based on platform behaviour. (E.g. x86 wants its 128-bit
SSE vectors to be 128-bit aligned.)

## Operations

CPU vendors usually offer "standard" C headers for their CPU specific
operations, such as [`arm_neon.h`][armneon] and [the `...mmintrin.h` headers for
x86(-64)][x86].

[armneon]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
[x86]: https://software.intel.com/sites/landingpage/IntrinsicsGuide

All of these would be exposed as compiler intrinsics with names very
similar to those that the vendor suggests (only difference would be
some form of manual namespacing, e.g. prefixing with the CPU target),
loadable via an `extern` block with an appropriate ABI. This subset of
intrinsics would be on the path to stabilisation (that is, one can
"import" them with `extern` in stable code), and would not be exported
by `std`.

Example:

```rust
extern "platform-intrinsic" {
    fn x86_mm_abs_epi16(a: Simd8<i16>) -> Simd8<i16>;
    // ...
}
```

These all use entirely concrete types, and this is the core interface
to these intrinsics: essentially it is just allowing code to exactly
specify a CPU instruction to use. These intrinsics only actually work
on a subset of the CPUs that Rust targets, and will result in compile
time errors if they are called on platforms that do not support
them. The signatures are typechecked, but in a "duck-typed" manner: it
will just ensure that the types are SIMD vectors with the appropriate
length and element type, it will not enforce a specific nominal type.

NB. The structural typing is just for the declaration: if a SIMD intrinsic
is declared to take a type `X`, it must always be called with `X`,
even if other types are structurally equal to `X`. Also, within a
signature, SIMD types that must be structurally equal must be nominally
equal. I.e. if the `add_...` all refer to the same intrinsic to add a
SIMD vector of bytes,

```rust
// (same length)
struct A(u8, u8, ..., u8);
struct B(u8, u8, ..., u8);

extern "platform-intrinsic" {
    fn add_aaa(x: A, y: A) -> A; // ok
    fn add_bbb(x: B, y: B) -> B; // ok
    fn add_aab(x: A, y: A) -> B; // error, expected B, found A
    fn add_bab(x: B, y: A) -> B; // error, expected A, found B
}

fn double_a(x: A) -> A {
    add_aaa(x, x)
}
fn double_b(x: B) -> B {
    add_aaa(x, x) // error, expected A, found B
}
```

There would additionally be a small set of cross-platform operations
that are either generally efficiently supported everywhere or are
extremely useful. These won't necessarily map to a single instruction,
but will be shimmed as efficiently as possible.

- shuffles and extracting/inserting elements
- comparisons
- arithmetic
- conversions

All of these intrinsics are imported via an `extern` directive similar
to the process for pre-existing intrinsics like `transmute`, however,
the SIMD operations are provided under a special ABI:
`platform-intrinsic`. Use of this ABI (and hence the intrinsics) is
initially feature-gated under the `platform_intrinsics` feature
name. Why `platform-intrinsic` rather than say `simd-intrinsic`? There
are non-SIMD platform-specific instructions that may be nice to expose
(for example, Intel defines an `_addcarry_u32` intrinsic corresponding
to the `ADC` instruction).

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
extern "platform-intrinsic" {
    fn simd_shuffle2<T, U>(v: T, w: T, idx: [i32; 2]) -> U;
    fn simd_shuffle4<T, U>(v: T, w: T, idx: [i32; 4]) -> U;
    fn simd_shuffle8<T, U>(v: T, w: T, idx: [i32; 8]) -> U;
    fn simd_shuffle16<T, U>(v: T, w: T, idx: [i32; 16]) -> U;
    // ...
}
```

The raw definitions are only checked for validity at monomorphisation
time, ensure that `T` and `U` are SIMD vector with the same element
type, `U` has the appropriate length etc. Libraries can use traits to
ensure that these will be enforced by the type checker too.

This approach has similar type "safety"/code-generation errors to the
vectors themselves.

These operations are semantically:

```rust
// vector of double length
let z = concat(v, w);

return [z[idx[0]], z[idx[1]], z[idx[2]], ...]
```

The index array `idx` has to be compile time constants. Out of bounds
indices yield errors.

Similarly, intrinsics for inserting/extracting elements into/out of
vectors are provided, to allow modelling the SIMD vectors as actual
CPU registers as much as possible:

```rust
extern "platform-intrinsic" {
    fn simd_insert<T, Elem>(v: T, i0: u32, elem: Elem) -> T;
    fn simd_extract<T, Elem>(v: T, i0: u32) -> Elem;
}
```

The `i0` indices do not have to be constant. These are equivalent to
`v[i0] = elem` and `v[i0]` respectively. They are type checked
similarly to the shuffles.

### Comparisons

Comparisons are implemented via intrinsics. The raw signatures would
look like:

```rust
extern "platform-intrinsic" {
    fn simd_eq<T, U>(v: T, w: T) -> U;
    fn simd_ne<T, U>(v: T, w: T) -> U;
    fn simd_lt<T, U>(v: T, w: T) -> U;
    fn simd_le<T, U>(v: T, w: T) -> U;
    fn simd_gt<T, U>(v: T, w: T) -> U;
    fn simd_ge<T, U>(v: T, w: T) -> U;
}
```

These are type checked during code-generation similarly to the
shuffles: ensuring that `T` and `U` have the same length, and that `U`
is appropriately "boolean"-y. Libraries can use traits to ensure that
these will be enforced by the type checker too.

### Arithmetic

Intrinsics will be provided for arithmetic operations like addition
and multiplication.

```rust
extern "platform-intrinsic" {
    fn simd_add<T>(x: T, y: T) -> T;
    fn simd_mul<T>(x: T, y: T) -> T;
    // ...
}
```

These will have codegen time checks that the element type is correct:

- `add`, `sub`, `mul`: any float or integer type
- `div`: any float type
- `and`, `or`, `xor`, `shl` (shift left), `shr` (shift right): any
  integer type

(The integer types are `i8`, ..., `i64`, `u8`, ..., `u64` and the
float types are `f32` and `f64`.)

### Why not inline asm?

One alternative to providing intrinsics is to instead just use
inline-asm to expose each CPU instruction. However, this approach has
essentially only one benefit (avoiding defining the intrinsics), but
several downsides, e.g.

- assembly is generally a black-box to optimisers, inhibiting
  optimisations, like algebraic simplification/transformation,
- programmers would have to manually synthesise the right sequence of
  operations to achieve a given shuffle, while having a generic
  shuffle intrinsic lets the compiler do it (NB. the intention is that
  the programmer will still have access to the platform specific
  operations for when the compiler synthesis isn't quite right),
- inline assembly is not currently stable in
  Rust and there's not a strong push for it to be so in the immediate
  future (although this could change).

Benefits of manual assembly writing, like instruction scheduling and
register allocation don't apply to the (generally) one-instruction
`asm!` blocks that replace the intrinsics (they need to be designed so
that the compiler has full control over register allocation, or else
the result will be strictly worse). Those possible advantages of hand
written assembly over intrinsics only come in to play when writing
longer blocks of raw assembly, i.e. some inner loop might be faster
when written as a single chunk of asm rather than as intrinsics.

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

With a `cfg_if!` macro that expands to the first `cfg` that is
satisfied (ala [@alexcrichton's `cfg-if`][cfg-if]), code might look
like:

[cfg-if]: https://crates.io/crates/cfg-if

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
  vector of pointers. This would require allowing
  pointers(/references?) in `repr(simd)` types.
- allow (and ignore for everything but type checking) zero-sized types
  in `repr(simd)` structs, to allow tagging them with markers
- the shuffle intrinsics could be made more relaxed in their type
  checking (i.e. not require that they return their second type
  parameter), to allow more type safety when combined with generic
  simd types:

      #[repr(simd)] struct Simd2<T>(T, T);
      extern "platform-intrinsic" {
          fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> Simd2<U>;
      }

  This should be a backwards-compatible generalisation.

# Alternatives

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
- With type-level integers, there could be one shuffle intrinsic:

     fn simd_shuffle<T, U, const N: usize>(x: T, y: T, idx: [u32; N]) -> U;

  NB. It is possible to add this as an additional intrinsic (possibly
  deprecating the `simd_shuffleNNN` forms) later.
- Type-level values can be applied more generally: since the shuffle
  indices have to be compile time constants, the shuffle could be

      fn simd_shuffle<T, U, const N: usize, const IDX: [u32; N]>(x: T, y: T) -> U;

- Instead of platform detection, there could be feature detection
  (e.g. "platform supports something equivalent to x86's `DPPS`"), but
  there probably aren't enough cross-platform commonalities for this
  to be worth it. (Each "feature" would essentially be a platform
  specific `cfg` anyway.)
- Check vector operators in debug mode just like the scalar versions.
- Make fixed length arrays `repr(simd)`-able (via just flattening), so
  that, say, `#[repr(simd)] struct u32x4([u32; 4]);` and
  `#[repr(simd)] struct f64x8([f64; 4], [f64; 4]);` etc works. This
  will be most useful if/when we allow generic-lengths, `#[repr(simd)]
  struct Simd<T, n>([T; n]);`
- have 100% guaranteed type-safety for generic `#[repr(simd)]` types
  and the generic intrinsics. This would probably require a relatively
  complicated set of traits (with compiler integration).

# Unresolved questions

- Should integer vectors get division automatically? Most CPUs
  don't support them for vectors.
- How should out-of-bounds shuffle and insert/extract indices be handled?
