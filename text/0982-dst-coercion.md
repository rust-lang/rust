- Feature Name: dst_coercions
- Start Date: 2015-03-16
- RFC PR: [rust-lang/rfcs#982](https://github.com/rust-lang/rfcs/pull/982)
- Rust Issue: [rust-lang/rust#18598](https://github.com/rust-lang/rust/issues/18598)

# Summary

Custom coercions allow smart pointers to fully participate in the DST system.
In particular, they allow practical use of `Rc<T>` and `Arc<T>` where `T` is unsized.

This RFC subsumes part of [RFC 401 coercions](https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md).

# Motivation

DST is not really finished without this, in particular there is a need for types
like reference counted trait objects (`Rc<Trait>`) which are not currently well-
supported (without coercions, it is pretty much impossible to create such values
with such a type).

# Detailed design

There is an `Unsize` trait and lang item. This trait signals that a type can be
converted using the compiler's coercion machinery from a sized to an unsized
type. All implementations of this trait are implicit and compiler generated. It
is an error to implement this trait. If `&T` can be coerced to `&U` then there
will be an implementation of `Unsize<U>` for `T`. E.g, `[i32; 42]:
Unsize<[i32]>`. Note that the existence of an `Unsize` impl does not signify a
coercion can itself can take place, it represents an internal part of the
coercion mechanism (it corresponds with `coerce_inner` from  RFC 401). The trait
is defined as:

```
#[lang="unsize"]
trait Unsize<T: ?Sized>: ::std::marker::PhantomFn<Self, T> {}
```

There are implementations for any fixed size array to the corresponding unsized
array, for any type to any trait that that type implements, for structs and
tuples where the last field can be unsized, and for any pair of traits where
`Self` is a sub-trait of `T` (see RFC 401 for more details).

There is a `CoerceUnsized` trait which is implemented by smart pointer types to
opt-in to DST coercions. It is defined as:

```
#[lang="coerce_unsized"]
trait CoerceUnsized<Target>: ::std::marker::PhantomFn<Self, Target> + Sized {}
```

An example implementation:

```
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<Rc<U>> for Rc<T> {}
impl<T: Zeroable+CoerceUnsized<U>, U: Zeroable> CoerceUnsized<NonZero<U>> for NonZero<T> {}

// For reference, the definitions of Rc and NonZero:
pub struct Rc<T: ?Sized> {
    _ptr: NonZero<*mut RcBox<T>>,
}
pub struct NonZero<T: Zeroable>(T);
```

Implementing `CoerceUnsized` indicates that the self type should be able to be
coerced to the `Target` type. E.g., the above implementation means that
`Rc<[i32; 42]>` can be coerced to `Rc<[i32]>`. There will be `CoerceUnsized` impls
for the various pointer kinds available in Rust and which allow coercions, therefore
`CoerceUnsized` when used as a bound indicates coercible types. E.g.,

```
fn foo<T: CoerceUnsized<U>, U>(x: T) -> U {
    x
}
```

Built-in pointer impls:

```
impl<'a, 'b: 'aT: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}

impl<'a, 'b: 'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
impl<'b, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'b T {}

impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}

impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}
```

Note that there are some coercions which are not given by `CoerceUnsized`, e.g.,
from safe to unsafe function pointers, so it really is a `CoerceUnsized` trait,
not a general `Coerce` trait.


## Compiler checking

### On encountering an implementation of `CoerceUnsized` (type collection phase)

* If the impl is for a built-in pointer type, we check nothing, otherwise...
* The compiler checks that the `Self` type is a struct or tuple struct and that
the `Target` type is a simple substitution of type parameters from the `Self`
type (i.e., That `Self` is `Foo<Ts>`, `Target` is `Foo<Us>` and that there exist
`Vs` and `Xs` (where `Xs` are all type parameters) such that `Target = [Vs/Xs]Self`.
One day, with HKT, this could be a regular part of type checking, for now
it must be an ad hoc check). We might enforce that this substitution is of the
form `X/Y` where `X` and `Y` are both formal type parameters of the
implementation (I don't think this is necessary, but it makes checking coercions
easier and is satisfied for all smart pointers).
* The compiler checks each field in the `Self` type against the corresponding field
in the `Target` type. Assuming `Fs` is the type of a field in `Self` and `Ft` is
the type of the corresponding field in `Target`, then either `Ft <: Fs` or
`Fs: CoerceUnsized<Ft>` (note that this includes some built-in coercions, coercions
unrelated to unsizing are excluded, these could probably be added later, if needed).
* There must be only one non-PhantomData field that is coerced.
* We record for each impl, the index of the field in the `Self` type which is
coerced.

### On encountering a potential coercion (type checking phase)

* If we have an expression with type `E` where the type `F` is required during
type checking and `E` is not a subtype of `F`, nor is it coercible using the
built-in coercions, then we search for a bound of `E: CoerceUnsized<F>`. Note
that we may not at this stage find the actual impl, but finding the bound is
good enough for type checking.

* If we require a coercion in the receiver of a method call or field lookup, we
perform the same search that we currently do, except that where we currently
check for coercions, we check for built-in coercions and then for `CoerceUnsized`
bounds. We must also check for `Unsize` bounds for the case where the receiver
is auto-deref'ed, but not autoref'ed.


### On encountering an adjustment (translation phase)

* In trans (which is post-monomorphisation) we should always be able to find an
impl for any `CoerceUnsized` bound.
* If the impl is for a built-in pointer type, then we use the current coercion
code for the various pointer kinds (`Box<T>` has different behaviour than `&` and
`*` pointers).
* Otherwise, we lookup which field is coerced due to the opt-in coercion, move
the object being coerced and coerce the field in question by recursing (the
built-in pointers are the base cases).


### Adjustment types

We add `AdjustCustom` to the `AutoAdjustment` enum as a placeholder for coercions
due to a `CoerceUnsized` bound. I don't think we need the `UnsizeKind` enum at
all now, since all checking is postponed until trans or relies on traits and impls.


# Drawbacks

Not as flexible as the previous proposal.

# Alternatives

The original [DST5 proposal](http://smallcultfollowing.com/babysteps/blog/2014/01/05/dst-take-5/)
contains a similar proposal with no opt-in trait, i.e., coercions are completely
automatic and arbitrarily deep. This is a little too magical and unpredicatable.
It violates some 'soft abstraction boundaries' by interefering with the deep
structure of objects, sometimes even automatically (and implicitly) allocating.

[RFC 401](https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md)
proposed a scheme for proposals where users write their own coercion using
intrinsics. Although more flexible, this allows for implcicit excecution of
arbitrary code. If we need the increased flexibility, I believe we can add a
manual option to the `CoerceUnsized` trait backwards compatibly.

The proposed design could be tweaked: for example, we could change the
`CoerceUnsized` trait in many ways (we experimented with an associated type to
indicate the field type which is coerced, for example).

# Unresolved questions

It is unclear to what extent DST coercions should support multiple fields that
refer to the same type parameter. `PhantomData<T>` should definitely be
supported as an "extra" field that's skipped, but can all zero-sized fields
be skipped? Are there cases where this would enable by-passing the abstractions
that make some API safe?

# Updates since being accepted

Since it was accepted, the RFC has been updated as follows:

1. `CoerceUnsized` was specified to ingore PhantomData fields.
