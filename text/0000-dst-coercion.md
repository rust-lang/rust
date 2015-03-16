- Feature Name: dst-coercions
- Start Date: 2015-03-16
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

// For reference, the definition of Rc:
pub struct Rc<T: ?Sized> {
    _ptr: NonZero<*mut RcBox<T>>,
}
```

Implementing `CoerceUnsized` indicates that the self type should be able to be
coerced to the `Target` type. E.g., the above implementation means that
`Rc<[i32; 42]>` can be coerced to `Rc<[i32]>`.


## Newtype coercions

We also add a new built-in coercion for 'newtype's. If `Foo<T>` is a tuple
struct with a single field with type `T` and `T` has at least the `?Sized`
bound, then coerce_inner(`Foo<T>`) = `Foo<U>` holds for any `T` and `U` where
`T` coerces to `U`.

This coercion is not opt-in. It is best thought of as an extension to the
coercion rule for structs with an unsized field, the extension is that here the
field conversion is a proper coercion, not an application of `coerce_inner`.
Note that this coercion can be recursively applied.


## Compiler checking

### On encountering an implementation of `CoerceUnsized` (type collection phase)

* The compiler checks that the `Self` type is a struct or tuple struct and that
the `Target` type is a simple substitution of type parameters from the `Self`
type (one day, with HKT, this could be a regular part of type checking, for now
it must be an ad hoc check). We might enforce that this substitution is of the
form `X/Y` where `X` and `Y` are both formal type parameters of the
implementation (I don't think this is necessary, but it makes checking coercions
easier and is satisfied for all smart pointers).
* The compiler checks each field in the `Self` type against the corresponding field
in the `Target` type. Either the field types must be subtypes or be coercible from the
`Self` field to the `Target` field (this is checked taking into account any
`Unsize` bounds in the environment which indicate that some coercion can take
place). Note that this per-field check uses only the built-in coercion
mechanics. It does not take into account `CoerceUnsized` impls (although we
might allow this in the future).
* There must be only one field that is coerced.
* We record in a side table a mapping from the impl to an adjustment. The
adjustment will contain the field which is coerced and a nested adjustment
representing that coercion. The nested adjustment will have a placeholder for
any use of the `Unsize` bound (we should require that there is exactly one such use).

### On encountering a potential coercion

* If we have an expression with type `E` where the type `F` is required during
type checking and `E` is not a subtype of `F`, nor is it coercible using the
built-in coercions, then we search for an implementation of `CoerceUnsized<F>`
for `E`. A match will give us a substitution of the formal type parameters of
the impl by some actual types.
* We look up the impl in the side table described above. The substitution is used
with the placeholder in the recorded adjustment to create a new coercion which
will map one field of the struct being coerced. That coercion should always be
valid (if it is not, there is a compiler bug).
* We create a new adjustment for the coerced expression. This will include the
index of the field which is deeply coerced and the adjustment for the coercion
described in the previous step.
* In trans, the adjustment is used to codegen a coercion by moving the coerced
value and changing the indicated field to a new type according to the nested
adjustment.

### Adjustment types

We add `AdjustCustom(usize, Box<AutoAdjustment>)` and
`AdjustNewtype(Box<AutoAdjustment>)` to the `AutoAdjustment` enum. These
represent the new custom and newtype coercions, respectively. We add
`UnsizePlaceHolder(Ty, Ty)` to the `UnsizeKind` enum to represent a placeholder
adjustment due to an `Unsize` bound.

### Example

For the above `Rc` impl, we record the following adjustment (with some trivial
bits and pieces elided):

```
AdjustCustom(0, AdjustNewType(
    AutoDerefRef {
        autoderefs: 1,
        autoref: AutoUnsafe(mut, AutoUnsize(
            UnsizeStruct(UnsizePlaceholder(T, U))))
    }))
```

When we need to coerce `Rc<[i32; 42]>` to `Rc<[i32]>`, we look up the impl and
find `T = [i32; 42]` and `U = [i32]` (note that we automatically require that
`Unsize` is satisfied when looking up the impl). We can therefore replace the
placeholder in the above adjustment with `UnsizeLength(42)`. That gives us the
real adjustment to store for trans.

# Drawbacks

Not as flexible as the previous proposal. Can't handle pointer-like types like
`Option<Box<T>>`.

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

The proposed design could be tweaked: we could make newtype coercions opt-in
(this would complicate other parts of the proposal though). We could change the
`CoerceUnsized` trait in many ways (we experimented with an associated type to
indicate the field type which is coerced, for example).

# Unresolved questions

None
