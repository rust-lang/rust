# Early and Late Bound Variables

In Rust, item definitions (like `fn`) can often have generic parameters, which
are always [_universally_ quantified][quant]. That is, if you have a function
like

```rust
fn foo<T>(x: T) { }
```

this function is defined "for all T" (not "for some specific T", which would be
[_existentially_ quantified][quant]).

[quant]: ./appendix/background.md#quantified

While Rust *items* can be quantified over types, lifetimes, and constants, the
types of values in Rust are only ever quantified over lifetimes. So you can
have a type like `for<'a> fn(&'a u32)`, which represents a function pointer
that takes a reference with any lifetime, or `for<'a> dyn Trait<'a>`, which is
a `dyn` trait for a trait implemented for any lifetime; but we have no type
like `for<T> fn(T)`, which would be a function that takes a value of *any type*
as a parameter. This is a consequence of monomorphization -- to support a value
of type `for<T> fn(T)`, we would need a single function pointer that can be
used for a parameter of any type, but in Rust we generate customized code for
each parameter type.

One consequence of this asymmetry is a weird split in how we represent some
generic types: _early-_ and _late-_ bound parameters.
Basically, if we cannot represent a type (e.g. a universally quantified type),
we have to bind it _early_ so that the unrepresentable type is never around.

Consider the following example:

```rust,ignore
fn foo<'a, 'b, T>(x: &'a u32, y: &'b T) where T: 'b { ... }
```

We cannot treat `'a`, `'b`, and `T` in the same way.  Types in Rust can't have
`for<T> { .. }`, only `for<'a> {...}`, so whenever you reference `foo` the type
you get back can't be `for<'a, 'b, T> fn(&'a u32, y: &'b T)`. Instead, the `T`
must be substituted early. In particular, you have:

```rust,ignore
let x = foo; // T, 'b have to be substituted here
x(...);      // 'a substituted here, at the point of call
x(...);      // 'a substituted here with a different value
```

## Early-bound parameters

Early-bound parameters in rustc are identified by an index, stored in the
[`ParamTy`] struct for types or the [`EarlyBoundRegion`] struct for lifetimes.
The index counts from the outermost declaration in scope. This means that as you
add more binders inside, the index doesn't change.

For example,

```rust,ignore
trait Foo<T> {
  type Bar<U> = (Self, T, U);
}
```

Here, the type `(Self, T, U)` would be `($0, $1, $2)`, where `$N` means a
[`ParamTy`] with the index of `N`.

In rustc, the [`Generics`] structure carries this information. So the
[`Generics`] for `Bar` above would be just like for `U` and would indicate the
'parent' generics of `Foo`, which declares `Self` and `T`.  You can read more
in [this chapter](./generics.md).

[`ParamTy`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamTy.html
[`EarlyBoundRegion`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.EarlyBoundRegion.html
[`Generics`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Generics.html

## Late-bound parameters

Late-bound parameters in `rustc` are handled quite differently (they are also
specialized to lifetimes since, right now, only late-bound lifetimes are
supported, though with GATs that has to change). We indicate their potential
presence by a [`Binder`] type. The [`Binder`] doesn't know how many variables
there are at that binding level. This can only be determined by walking the
type itself and collecting them. So a type like `for<'a, 'b> ('a, 'b)` would be
`for (^0.a, ^0.b)`. Here, we just write `for` because we don't know the names
of the things bound within.

Moreover, a reference to a late-bound lifetime is written `^0.a`:

- The `0` is the index; it identifies that this lifetime is bound in the
  innermost binder (the `for`).
- The `a` is the "name"; late-bound lifetimes in rustc are identified by a
  "name" -- the [`BoundRegionKind`] enum. This enum can contain a
  [`DefId`][defid] or it might have various "anonymous" numbered names. The
  latter arise from types like `fn(&u32, &u32)`, which are equivalent to
  something like `for<'a, 'b> fn(&'a u32, &'b u32)`, but the names of those
  lifetimes must be generated.

This setup of not knowing the full set of variables at a binding level has some
advantages and some disadvantages. The disadvantage is that you must walk the
type to find out what is bound at the given level and so forth. The advantage
is primarily that, when constructing types from Rust syntax, if we encounter
anonymous regions like in `fn(&u32)`, we just create a fresh index and don't have
to update the binder.

[`Binder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Binder.html
[`BoundRegionKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.BoundRegionKind.html
[defid]: ./hir.html#identifiers-in-the-hir
