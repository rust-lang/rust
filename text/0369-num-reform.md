- Start Date: 2014-09-16
- RFC PR: [rust-lang/rfcs#369](https://github.com/rust-lang/rfcs/pull/369)
- Rust Issue: [rust-lang/rust#18640](https://github.com/rust-lang/rust/issues/18640)

# Summary

This RFC is preparation for API stabilization for the `std::num` module.  The
proposal is to finish the simplification efforts started in
[@bjz's reversal of the numerics hierarcy](https://github.com/rust-lang/rust/issues/10387).

Broadly, the proposal is to collapse the remaining numeric hierarchy
in `std::num`, and to provide only limited support for generic
programming (roughly, only over primitive numeric types that vary
based on size).  Traits giving detailed numeric hierarchy can and
should be provided separately through the Cargo ecosystem.

Thus, this RFC proposes to flatten or remove most of the traits
currently provided by `std::num`, and generally to simplify the module
as much as possible in preparation for API stabilization.

# Motivation

## History

Starting in early 2013, there was
[an effort](https://github.com/rust-lang/rust/issues/4819) to design a
comprehensive "numeric hierarchy" for Rust: a collection of traits classifying a
wide variety of numbers and other algebraic objects. The intent was to allow
highly-generic code to be written for algebraic structures and then instantiated
to particular types.

This hierarchy covered structures like bigints, but also primitive integer and
float types. It was an enormous and long-running community effort.

Later, [it was recognized](https://github.com/rust-lang/rust/issues/10387) that
building such a hierarchy within `libstd` was misguided:

> @bjz The API that resulted from #4819 attempted, like Haskell, to blend both
> the primitive numerics and higher level mathematical concepts into one
> API. This resulted in an ugly hybrid where neither goal was adequately met. I
> think the libstd should have a strong focus on implementing fundamental
> operations for the base numeric types, but no more. Leave the higher level
> concepts to libnum or future community projects.

The `std::num` module has thus been slowly migrating *away* from a large trait
hierarchy toward a simpler one providing just APIs for primitive data types:
this is
[@bjz's reversal of the numerics hierarcy](https://github.com/rust-lang/rust/issues/10387).

Along side this effort, there are already external numerics packages like
[@bjz's num-rs](https://github.com/bjz/num-rs).

But we're not finished yet.

## The current state of affairs

The `std::num` module still contains quite a few traits that subdivide out
various features of numbers:

```rust
pub trait Zero: Add<Self, Self> {
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

pub trait One: Mul<Self, Self> {
    fn one() -> Self;
}

pub trait Signed: Num + Neg<Self> {
    fn abs(&self) -> Self;
    fn abs_sub(&self, other: &Self) -> Self;
    fn signum(&self) -> Self;
    fn is_positive(&self) -> bool;
    fn is_negative(&self) -> bool;
}

pub trait Unsigned: Num {}

pub trait Bounded {
    fn min_value() -> Self;
    fn max_value() -> Self;
}

pub trait Primitive: Copy + Clone + Num + NumCast + PartialOrd + Bounded {}

pub trait Num: PartialEq + Zero + One + Neg<Self> + Add<Self,Self> + Sub<Self,Self>
             + Mul<Self,Self> + Div<Self,Self> + Rem<Self,Self> {}

pub trait Int: Primitive + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv
             + Bounded + Not<Self> + BitAnd<Self,Self> + BitOr<Self,Self>
             + BitXor<Self,Self> + Shl<uint,Self> + Shr<uint,Self> {
    fn count_ones(self) -> uint;
    fn count_zeros(self) -> uint { ... }
    fn leading_zeros(self) -> uint;
    fn trailing_zeros(self) -> uint;
    fn rotate_left(self, n: uint) -> Self;
    fn rotate_right(self, n: uint) -> Self;
    fn swap_bytes(self) -> Self;
    fn from_be(x: Self) -> Self { ... }
    fn from_le(x: Self) -> Self { ... }
    fn to_be(self) -> Self { ... }
    fn to_le(self) -> Self { ... }
}

pub trait FromPrimitive {
    fn from_i64(n: i64) -> Option<Self>;
    fn from_u64(n: u64) -> Option<Self>;

    // many additional defaulted methods
    // ...
}

pub trait ToPrimitive {
    fn to_i64(&self) -> Option<i64>;
    fn to_u64(&self) -> Option<u64>;

    // many additional defaulted methods
    // ...
}

pub trait NumCast: ToPrimitive {
    fn from<T: ToPrimitive>(n: T) -> Option<Self>;
}

pub trait Saturating {
    fn saturating_add(self, v: Self) -> Self;
    fn saturating_sub(self, v: Self) -> Self;
}

pub trait CheckedAdd: Add<Self, Self> {
    fn checked_add(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedSub: Sub<Self, Self> {
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedMul: Mul<Self, Self> {
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

pub trait CheckedDiv: Div<Self, Self> {
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

pub trait Float: Signed + Primitive {
    // a huge collection of static functions (for constants) and methods
    ...
}

pub trait FloatMath: Float {
    // an additional collection of methods
}
```

The `Primitive` traits are intended primarily to support a mechanism,
`#[deriving(FromPrimitive)]`, that makes it easy to provide
conversions from numeric types to C-like `enum`s.

The `Saturating` and `Checked` traits provide operations that provide
special handling for overflow and other numeric errors.

Almost all of these traits are currently included in the prelude.

In addition to these traits, the `std::num` module includes a couple
dozen free functions, most of which duplicate methods available though
traits.

## Where we want to go: a summary

The goal of this RFC is to refactor the `std::num` hierarchy with the
following goals in mind:

* Simplicity.

* *Limited* generic programming: being able to work generically over
  the natural classes of *primitive* numeric types that vary only by
  size.  There should be enough abstraction to support porting
  `strconv`, the generic string/number conversion code used in `std`.

* Minimizing dependencies for `libcore`. For example, it should not
  require `cmath`.

* Future-proofing for external numerics packages. The Cargo ecosystem
  should ultimately provide choices of sophisticated numeric
  hierarchies, and `std::num` should not get in the way.

# Detailed design

## Overview: the new hierarchy

This RFC proposes to collapse the trait hierarchy in `std::num` to
just the following traits:

* `Int`, implemented by all primitive integer types (`u8` - `u64`, `i8`-`i64`)
    * `UnsignedInt`, implemented by `u8` - `u64`
* `Signed`, implemented by all signed primitive numeric types (`i8`-`i64`, `f32`-`f64`)
* `Float`, implemented by `f32` and `f64`
    * `FloatMath`, implemented by `f32` and `f64`, which provides functionality from `cmath`

These traits inherit from all applicable overloaded operator traits
(from `core::ops`).  They suffice for generic programming over several
basic categories of primitive numeric types.

As designed, these traits include a certain amount of redundancy
between `Int` and `Float`. The Alternatives section shows how this
could be factored out into a separate `Num` trait. But doing so
suggests a level of generic programming that these traits aren't
intended to support.

The main reason to pull out `Signed` into its own trait is so that it
can be added to the prelude. (Further discussion below.)

## Detailed definitions

Below is the full definition of these traits. The functionality
remains largely as it is today, just organized into fewer traits:

```rust
pub trait Int: Copy + Clone + PartialOrd + PartialEq
             + Add<Self,Self> + Sub<Self,Self>
             + Mul<Self,Self> + Div<Self,Self> + Rem<Self,Self>
             + Not<Self> + BitAnd<Self,Self> + BitOr<Self,Self>
             + BitXor<Self,Self> + Shl<uint,Self> + Shr<uint,Self>
{
    // Constants
    fn zero() -> Self;  // These should be associated constants when those are available
    fn one() -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;

    // Deprecated:
    // fn is_zero(&self) -> bool;

    // Bit twidling
    fn count_ones(self) -> uint;
    fn count_zeros(self) -> uint { ... }
    fn leading_zeros(self) -> uint;
    fn trailing_zeros(self) -> uint;
    fn rotate_left(self, n: uint) -> Self;
    fn rotate_right(self, n: uint) -> Self;
    fn swap_bytes(self) -> Self;
    fn from_be(x: Self) -> Self { ... }
    fn from_le(x: Self) -> Self { ... }
    fn to_be(self) -> Self { ... }
    fn to_le(self) -> Self { ... }

    // Checked arithmetic
    fn checked_add(self, v: Self) -> Option<Self>;
    fn checked_sub(self, v: Self) -> Option<Self>;
    fn checked_mul(self, v: Self) -> Option<Self>;
    fn checked_div(self, v: Self) -> Option<Self>;
    fn saturating_add(self, v: Self) -> Self;
    fn saturating_sub(self, v: Self) -> Self;
}

pub trait UnsignedInt: Int {
    fn is_power_of_two(self) -> bool;
    fn checked_next_power_of_two(self) -> Option<Self>;
    fn next_power_of_two(self) -> Self;
}

pub trait Signed: Neg<Self> {
    fn abs(&self) -> Self;
    fn signum(&self) -> Self;
    fn is_positive(&self) -> bool;
    fn is_negative(&self) -> bool;

    // Deprecated:
    // fn abs_sub(&self, other: &Self) -> Self;
}

pub trait Float: Copy + Clone + PartialOrd + PartialEq + Signed
               + Add<Self,Self> + Sub<Self,Self>
               + Mul<Self,Self> + Div<Self,Self> + Rem<Self,Self>
{
    // Constants
    fn zero() -> Self;  // These should be associated constants when those are available
    fn one() -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;

    // Classification and decomposition
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_normal(self) -> bool;
    fn classify(self) -> FPCategory;
    fn integer_decode(self) -> (u64, i16, i8);

    // Float intrinsics
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;

    // Conveniences
    fn fract(self) -> Self;
    fn recip(self) -> Self;
    fn rsqrt(self) -> Self;
    fn to_degrees(self) -> Self;
    fn to_radians(self) -> Self;
    fn log(self, base: Self) -> Self;
}

// This lives directly in `std::num`, not `core::num`, since it requires `cmath`
pub trait FloatMath: Float {
    // Exactly the methods defined in today's version
}
```

## Float constants, float math, and `cmath`

This RFC proposes to:

* Remove all float constants from the `Float` trait. These constants
  are available directly from the `f32` and `f64` modules, and are not
  really useful for the kind of generic programming these new traits
  are intended to allow.

* Continue providing various `cmath` functions as methods in the
  `FloatMath` trait. Putting this in a separate trait means that
  `libstd` depends on `cmath` but `libcore` does not.

## Free functions

All of the free functions defined in `std::num` are deprecated.

## The prelude

The prelude will only include the `Signed` trait, as the operations it
provides are widely expected to be available when they apply.

The reason for removing the rest of the traits is two-fold:

* The remaining operations are relatively uncommon. Note that various
  overloaded operators, like `+`, work regardless of this choice.
  Those doing intensive work with e.g. floats would only need to
  import `Float` and `FloatMath`.

* Keeping this functionality out of the prelude means that the names
  of methods and associated items remain available for external
  numerics libraries in the Cargo ecosystem.

## `strconv`, `FromStr`, `ToStr`, `FromStrRadix`, `ToStrRadix`

Currently, traits for converting from `&str` and to `String` are both
included, in their own modules, in `libstd`. This is largely due to
the desire to provide `impl`s for numeric types, which in turn relies
on `std::num::strconv`.

This RFC proposes to:

* Move the `FromStr` trait into `core::str`.
* Rename the `ToStr` trait to `ToString`, and move it to `collections::string`.
* Break up and revise `std::num::strconv` into separate, *private*
  modules that provide the needed functionality for the `from_str` and
  `to_string` methods. (Some of this functionality has already
  migrated to `fmt` and been deprecated in `strconv`.)
* Move the `FromStrRadix` into `core::num`.
* Remove `ToStrRadix`, which is already deprecated in favor of `fmt`.

## `FromPrimitive` and friends

Ideally, the `FromPrimitive`, `ToPrimitive`, `Primitive`, `NumCast`
traits would all be removed in favor of a more principled way of
working with C-like enums. However, such a replacement is outside of
the scope of this RFC, so these traits are left (as `#[experimental]`)
for now. A follow-up RFC proposing a better solution should appear soon.

In the meantime, see
[this proposal](https://github.com/rust-lang/rust/issues/10418) and
the discussion on
[this issue](https://github.com/rust-lang/rust/issues/10272) about
`Ordinal` for the rough direction forward.

# Drawbacks

This RFC somewhat reduces the potential for writing generic numeric
code with `std::num` traits. This is intentional, however: the new
design represents "just enough" generics to cover differently-sized
built-in types, without any attempt at general algebraic abstraction.

# Alternatives

The status quo is clearly not ideal, and as explained above there was
a long attempt at providing a more complete numeric hierarchy in `std`.
So *some* collapse of the hierarchy seems desirable.

That said, there are other possible factorings. We could introduce the
following `Num` trait to factor out commonalities between `Int` and `Float`:

```rust
pub trait Num: Copy + Clone + PartialOrd + PartialEq
             + Add<Self,Self> + Sub<Self,Self>
             + Mul<Self,Self> + Div<Self,Self> + Rem<Self,Self>
{
    fn zero() -> Self;  // These should be associated constants when those are available
    fn one() -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
}
```

However, it's not clear whether this factoring is worth having a more
complex hierarchy, especially because the traits are not intended for
generic programming at that level (and generic programming across
integer and floating-point types is likely to be extremely rare)

The signed and unsigned operations could be offered on more types,
allowing removal of more traits but a less clear-cut semantics.


# Unresolved questions

This RFC does not propose a replacement for
`#[deriving(FromPrimitive)]`, leaving the relevant traits in limbo
status. (See
[this proposal](https://github.com/rust-lang/rust/issues/10418) and
the discussion on
[this issue](https://github.com/rust-lang/rust/issues/10272) about
`Ordinal` for the rough direction forward.)
