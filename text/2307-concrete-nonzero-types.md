- Feature Name: `concrete-nonzero-types`
- Start Date: 2018-01-21
- RFC PR: [rust-lang/rfcs#2307](https://github.com/rust-lang/rfcs/pull/2307)
- Rust Issue: [rust-lang/rust#49137](https://github.com/rust-lang/rust/issues/49137)

# Summary
[summary]: #summary

Add `std::num::NonZeroU32` and eleven other concrete types (one for each primitive integer type)
to replace and deprecate `core::nonzero::NonZero<T>`.
(Non-zero/non-null raw pointers are available through
[`std::ptr::NonNull<U>`](https://doc.rust-lang.org/nightly/std/ptr/struct.NonNull.html).)

# Background
[background]: #background

The `&T` and `&mut T` types are represented in memory as pointers,
and the type system ensures that they’re always valid.
In particular, they can never be NULL.
Since at least 2013, rustc has taken advantage of that fact to optimize the memory representation
of `Option<&T>` and `Option<&mut T>` to be the same as `&T` and `&mut T`,
with the forbidden NULL value indicating `Option::None`.

Later (still before Rust 1.0),
a `core::nonzero::NonZero<T>` generic wrapper type was added to extend this optimization
to raw pointers (as used in types like `Box<T>` or `Vec<T>`) and integers,
encoding in the type system that they can not be null/zero.
Its API today is:

```rust
#[lang = "non_zero"]
#[unstable]
pub struct NonZero<T: Zeroable>(T);

#[unstable]
impl<T: Zeroable> NonZero<T> {
    pub const unsafe fn new_unchecked(x: T) -> Self { NonZero(x) }
    pub fn new(x: T) -> Option<Self> { if x.is_zero() { None } else { Some(NonZero(x)) }}
    pub fn get(self) -> T { self.0 }
}

#[unstable]
pub unsafe trait Zeroable {
    fn is_zero(&self) -> bool;
}

impl Zeroable for /* {{i,u}{8, 16, 32, 64, 128, size}, *{const,mut} T where T: ?Sized} */
```

The tracking issue for these unstable APIs is
[rust#27730](https://github.com/rust-lang/rust/issues/27730).

[`std::ptr::NonNull`](https://doc.rust-lang.org/nightly/std/ptr/struct.NonNull.html)
was stabilized in [in Rust 1.25](https://github.com/rust-lang/rust/pull/46952),
wrapping `NonZero` further for raw pointers and adding pointer-specific APIs.

# Motivation
[motivation]: #motivation

With `NonNull` covering pointers, the remaining use cases for `NonZero` are integers.

One problem of the current API is that
it is unclear what happens or what *should* happen to `NonZero<T>` or `Option<NonZero<T>>`
when `T` is some type other than a raw pointer or a primitive integer.
In particular, crates outside of `std` can implement `Zeroable` for their abitrary types
since it is a public trait.

To avoid this question entirely,
this RFC proposes replacing the generic type and trait with twelve concrete types in `std::num`,
one for each primitive integer type.
This is similar to the existing atomic integer types like `std::sync::atomic::AtomicU32`.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

When an integer value can never be zero because of the way an algorithm works,
this fact can be encoded in the type system
by using for example the `NonZeroU32` type instead of `u32`.

This enables code recieving such a value to safely make some assuptions,
for example that dividing by this value will not cause a `attempt to divide by zero` panic.
This may also enable the compiler to make some memory optimizations,
for example `Option<NonZeroU32>` might take no more space than `u32`
(with `None` represented as zero).

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

A new private `macro_rules!` macro is defined and used in `core::num` that expands to
twelve sets of items like below, one for each of:

* `u8`
* `u16`
* `u32`
* `u64`
* `u128`
* `usize`
* `i8`
* `i16`
* `i32`
* `i64`
* `i128`
* `isize`

These types are also re-exported in `std::num`.

```rust
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NonZeroU32(NonZero<u32>);

impl NonZeroU32 {
    pub const unsafe fn new_unchecked(n: u32) -> Self { Self(NonZero(n)) }
    pub fn new(n: u32) -> Option<Self> { if n == 0 { None } else { Some(Self(NonZero(n))) }}
    pub fn get(self) -> u32 { self.0.0 }
}

impl fmt::Debug for NonZeroU32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.get(), f)
    }
}

// Similar impls for Display, Binary, Octal, LowerHex, and UpperHex
```

Additionally, the `core::nonzero` module and its contents (`NonZero` and `Zeroable`)
are deprecated with a warning message that suggests using `ptr::NonNull` or `num::NonZero*` instead.

A couple release cycles later, the module is made private to libcore and reduced to:

```rust
/// Implementation detail of `ptr::NonNull` and `num::NonZero*`
#[lang = "non_zero"]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct NonZero(pub(crate) T);

impl<T: CoerceUnsized<U>> CoerceUnsized<NonZero<U>> for NonZero<T> {}
```

The memory layout of `Option<&T>` is a
[documented](https://doc.rust-lang.org/nomicon/other-reprs.html#reprc)
guarantee of the Rust language.
This RFC does **not** propose extending this guarantee to these new types.
For example, `size_of::<Option<NonZeroU32>>() == size_of::<NonZeroU32>()` may or may not be true.
It happens to be in current rustc,
but an alternative Rust implementation could define `num::NonZero*` purely as library types.

# Drawbacks
[drawbacks]: #drawbacks

This adds to the ever-expanding API surface of the standard library.

# Rationale and alternatives
[alternatives]: #alternatives

* Memory layout optimization for non-zero integers mostly exist in rustc today
  because their implementation is very close (or the same) as for non-null pointers.
  But maybe they’re not useful enough to justify any dedicated public API.
  `core::nonzero` could be deprecated and made private without adding `num::NonZero*`,
  with only `ptr::NonNull` exposing such functionality.

* On the other hand,
  maybe zero is “less special” for integers than NULL is for pointers.
  Maybe instead of `num::NonZero*` we should consider some other feature
  to enable creating integer wrapper types that restrict values to an arbitrary sub-range
  (making this known to the compiler for memory layout optimizations),
  similar to how [PR #45225](https://github.com/rust-lang/rust/pull/45225)
  restricts the primitive type `char` to `0 ..= 0x10FFFF`.
  Making entire bits available unlocks more potential future optimizations than a single value.

  However no design for such a feature has been proposed, whereas `NonZero` is already implemented.
  The author’s position is that `num::NonZero*` should be added
  as it is still useful and can be stabilized such sooner,
  and it does not prevent adding another language feature later.

* In response to “what if `Zeroable` is implemented for other types”
  it was suggested to prevent such `impl`s by making the trait permanently-unstable,
  or effectively private (by moving it in a private module
  and keeping it `pub trait` to fool the *private in public* lint).
  The author feels that such abuses of the stability or privacy systems
  do not belong in stable APIs.
  (Stable APIs that mention traits like `RangeArgument` that are not stable *yet*
  but have a path to stabilization are less of an abuse.)

* Still, we could decide on some answer to “`Zeroable` for abitrary types”,
  implement and test it, stabilize `NonZero<T>` and `Zeroable` as-is
  (re-exported in `std`), and not add `num::NonZero*`.

* Instead of `std::num` the new types could be in some other location,
  such as the modules named after their respective primitive types.
  For example `std::u32::NonZeroU32` or `std::u32::NonZero`.
  The former looks redundant,
  and the latter might lead to code that looks ambiguous if the type itself is imported
  instead of importing the module and using a qualified `u32::NonZero` path.

* We could drop the `NonZeroI*` wrappers for signed integers.
  They’re included in this RFC because it’s easy,
  but every use of non-zero integers the author has seen so far has been with unsigned ones.
  This would cut the number of new types from 12 to 6.

# Unresolved questions
[unresolved]: #unresolved-questions

Should the memory layout of e.g. `Option<NonZeroU32>` be a language guarantee?

Discussion of the design of a new language feature
for integer types restricted to an arbitrary sub-range (see second unresolved question)
is out of scope for this RFC.
Discussing the potential existence of such a feature
as a reason **not** to add non-zero integer types is in scope.
