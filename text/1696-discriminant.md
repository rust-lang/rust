- Feature Name: discriminant
- Start Date: 2016-08-01
- RFC PR: https://github.com/rust-lang/rfcs/pull/1696
- Rust Issue: [#24263](https://github.com/rust-lang/rust/pull/24263), [#34785](https://github.com/rust-lang/rust/pull/34785)

# Summary
[summary]: #summary

Add a function that extracts the discriminant from an enum variant as a comparable, hashable, printable, but (for now) opaque and unorderable type.

# Motivation
[motivation]: #motivation

When using an ADT enum that contains data in some of the variants, it is sometimes desirable to know the variant but ignore the data, in order to compare two values by variant or store variants in a hash map when the data is either unhashable or unimportant.

The motivation for this is mostly identical to [RFC 639](https://github.com/rust-lang/rfcs/blob/master/text/0639-discriminant-intrinsic.md#motivation).

# Detailed design
[design]: #detailed-design

The proposed design has been implemented at [#34785](https://github.com/rust-lang/rust/pull/34785) (after some back-and-forth). That implementation is copied at the end of this section for reference.

A struct `Discriminant<T>` and a free function `fn discriminant<T>(v: &T) -> Discriminant<T>` are added to `std::mem` (for lack of a better home, and noting that `std::mem` already contains similar parametricity escape hatches such as `size_of`). For now, the `Discriminant` struct is simply a newtype over `u64`, because that's what the `discriminant_value` intrinsic returns, and a `PhantomData` to allow it to be generic over `T`.

Making `Discriminant` generic provides several benefits:

- `discriminant(&EnumA::Variant) == discriminant(&EnumB::Variant)` is statically prevented.
- In the future, we can implement different behavior for different kinds of enums. For example, if we add a way to distinguish C-like enums at the type level, then we can add a method like `Discriminant::into_inner` for only those enums. Or enums with certain kinds of discriminants could become orderable.

The function no longer requires a `Reflect` bound on its argument even though discriminant extraction is a partial violation of parametricity, in that a generic function with no bounds on its type parameters can nonetheless find out some information about the input types, or perform a "partial equality" comparison. This is debatable (see [this comment](https://github.com/rust-lang/rfcs/pull/639#issuecomment-86441840), [this comment](https://github.com/rust-lang/rfcs/pull/1696#issuecomment-236669066) and open question #2), especially in light of specialization. The situation is comparable to `TypeId::of` (which requires the bound) and `mem::size_of_val` (which does not). Note that including a bound is the conservative decision, because it can be backwards-compatibly removed.

```rust
/// Returns a value uniquely identifying the enum variant in `v`.
///
/// If `T` is not an enum, calling this function will not result in undefined behavior, but the
/// return value is unspecified.
///
/// # Stability
///
/// Discriminants can change if enum variants are reordered, if a new variant is added
/// in the middle, or (in the case of a C-like enum) if explicitly set discriminants are changed.
/// Therefore, relying on the discriminants of enums outside of your crate may be a poor decision.
/// However, discriminants of an identical enum should not change between minor versions of the
/// same compiler.
///
/// # Examples
///
/// This can be used to compare enums that carry data, while disregarding
/// the actual data:
///
/// ```
/// #![feature(discriminant_value)]
/// use std::mem;
///
/// enum Foo { A(&'static str), B(i32), C(i32) }
///
/// assert!(mem::discriminant(&Foo::A("bar")) == mem::discriminant(&Foo::A("baz")));
/// assert!(mem::discriminant(&Foo::B(1))     == mem::discriminant(&Foo::B(2)));
/// assert!(mem::discriminant(&Foo::B(3))     != mem::discriminant(&Foo::C(3)));
/// ```
pub fn discriminant<T>(v: &T) -> Discriminant<T> {
    unsafe {
        Discriminant(intrinsics::discriminant_value(v), PhantomData)
    }
}

/// Opaque type representing the discriminant of an enum.
///
/// See the `discriminant` function in this module for more information.
pub struct Discriminant<T>(u64, PhantomData<*const T>);

impl<T> Copy for Discriminant<T> {}

impl<T> clone::Clone for Discriminant<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> cmp::PartialEq for Discriminant<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl<T> cmp::Eq for Discriminant<T> {}

impl<T> hash::Hash for Discriminant<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> fmt::Debug for Discriminant<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(fmt)
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

1. Anytime we reveal more details about the memory representation of a `repr(rust)` type, we add back-compat guarantees. The author is of the opinion that the proposed `Discriminant` newtype still hides enough to mitigate this drawback. (But see open question #1.)
2. Adding another function and type to core implies an additional maintenance burden, especially when more enum layout optimizations come around (however, there is hardly any burden on top of that associated with the extant `discriminant_value` intrinsic).

# Alternatives
[alternatives]: #alternatives

1. Do nothing: there is no stable way to extract the discriminant from an enum variant. Users who need such a feature will need to write (or generate) big match statements and hope they optimize well (this has been servo's approach).
2. Directly stabilize the `discriminant_value` intrinsic, or a wrapper that doesn't use an opaque newtype. This more drastically precludes future enum representation optimizations, and won't be able to take advantage of future type system improvements that would let `discriminant` return a type dependent on the enum.

# Unresolved questions
[unresolved]: #unresolved-questions

1. Can the return value of `discriminant(&x)` be considered stable between subsequent compilations of the same code? How about if the enum in question is changed by modifying a variant's name? by adding a variant?
2. Is the `T: Reflect` bound necessary?
3. Can `Discriminant` implement `PartialOrd`?
