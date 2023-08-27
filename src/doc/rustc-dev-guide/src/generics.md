# Generics and GenericArgs

Given a generic type `MyType<A, B, …>`, we may want to swap out the generics `A, B, …` for some
other types (possibly other generics or concrete types). We do this a lot while doing type
inference, type checking, and trait solving. Conceptually, during these routines, we may find out
that one type is equal to another type and want to swap one out for the other and then swap that out
for another type and so on until we eventually get some concrete types (or an error).

In rustc this is done using [GenericArgsRef].
Conceptually, you can think of `GenericArgsRef` as a list of types that are to be substituted for
 the generic type parameters of the ADT.

`GenericArgsRef` is a type alias of `&'tcx List<GenericArg<'tcx>>` (see [`List` rustdocs][list]).
[`GenericArg`] is essentially a space-efficient wrapper around [`GenericArgKind`], which is an enum
indicating what kind of generic the type parameter is (type, lifetime, or const).
Thus, `GenericArgsRef` is conceptually like a `&'tcx [GenericArgKind<'tcx>]` slice (but it is
actually a `List`).

[list]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.List.html
[`GenericArg`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.GenericArg.html
[`GenericArgKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.GenericArgKind.html
[GenericArgsRef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.GenericArgsRef.html

So why do we use this `List` type instead of making it really a slice? It has the length "inline",
so `&List` is only 32 bits. As a consequence, it cannot be "subsliced" (that only works if the
length is out of line).

This also implies that you can check two `List`s for equality via `==` (which would be not be
possible for ordinary slices). This is precisely because they never represent a "sub-list", only the
complete `List`, which has been hashed and interned.

So pulling it all together, let’s go back to our example above:

```rust,ignore
struct MyStruct<T>
```

- There would be an `AdtDef` (and corresponding `DefId`) for `MyStruct`.
- There would be a `TyKind::Param` (and corresponding `DefId`) for `T` (more later).
- There would be a `GenericArgsRef` containing the list `[GenericArgKind::Type(Ty(T))]`
    - The `Ty(T)` here is my shorthand for entire other `ty::Ty` that has `TyKind::Param`, which we
      mentioned in the previous point.
- This is one `TyKind::Adt` containing the `AdtDef` of `MyStruct` with the `GenericArgsRef` above.

Finally, we will quickly mention the
[`Generics`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Generics.html) type. It
is used to give information about the type parameters of a type.

### Unsubstituted Generics

So above, recall that in our example the `MyStruct` struct had a generic type `T`. When we are (for
example) type checking functions that use `MyStruct`, we will need to be able to refer to this type
`T` without actually knowing what it is. In general, this is true inside all generic definitions: we
need to be able to work with unknown types. This is done via `TyKind::Param` (which we mentioned in
the example above).

Each `TyKind::Param` contains two things: the name and the index. In general, the index fully
defines the parameter and is used by most of the code. The name is included for debug print-outs.
There are two reasons for this. First, the index is convenient, it allows you to include into the
list of generic arguments when substituting. Second, the index is more robust. For example, you
could in principle have two distinct type parameters that use the same name, e.g. `impl<A> Foo<A> {
fn bar<A>() { .. } }`, although the rules against shadowing make this difficult (but those language
rules could change in the future).

The index of the type parameter is an integer indicating its order in the list of the type
parameters. Moreover, we consider the list to include all of the type parameters from outer scopes.
Consider the following example:

```rust,ignore
struct Foo<A, B> {
  // A would have index 0
  // B would have index 1

  .. // some fields
}
impl<X, Y> Foo<X, Y> {
  fn method<Z>() {
    // inside here, X, Y and Z are all in scope
    // X has index 0
    // Y has index 1
    // Z has index 2
  }
}
```

When we are working inside the generic definition, we will use `TyKind::Param` just like any other
`TyKind`; it is just a type after all. However, if we want to use the generic type somewhere, then
we will need to do substitutions.

For example suppose that the `Foo<A, B>` type from the previous example has a field that is a
`Vec<A>`. Observe that `Vec` is also a generic type. We want to tell the compiler that the type
parameter of `Vec` should be replaced with the `A` type parameter of `Foo<A, B>`. We do that with
substitutions:

```rust,ignore
struct Foo<A, B> { // Adt(Foo, &[Param(0), Param(1)])
  x: Vec<A>, // Adt(Vec, &[Param(0)])
  ..
}

fn bar(foo: Foo<u32, f32>) { // Adt(Foo, &[u32, f32])
  let y = foo.x; // Vec<Param(0)> => Vec<u32>
}
```

This example has a few different substitutions:

- In the definition of `Foo`, in the type of the field `x`, we replace `Vec`'s type parameter with
  `Param(0)`, the first parameter of `Foo<A, B>`, so that the type of `x` is `Vec<A>`.
- In the function `bar`, we specify that we want a `Foo<u32, f32>`. This means that we will
  substitute `Param(0)` and `Param(1)` with `u32` and `f32`.
- In the body of `bar`, we access `foo.x`, which has type `Vec<Param(0)>`, but `Param(0)` has been
  substituted for `u32`, so `foo.x` has type `Vec<u32>`.

Let’s look a bit more closely at that last substitution to see why we use indexes. If we want to
find the type of `foo.x`, we can get generic type of `x`, which is `Vec<Param(0)>`. Now we can take
the index `0` and use it to find the right type substitution: looking at `Foo`'s `GenericArgsRef`,
we have the list `[u32, f32]` , since we want to replace index `0`, we take the 0-th index of this
list, which is `u32`. Voila!

You may have a couple of followup questions…

 **`type_of`** How do we get the "generic type of `x`"? You can get the type of pretty much anything
 with the   `tcx.type_of(def_id)` query. In this case, we would pass the `DefId` of the field `x`.
 The `type_of` query always returns the definition with the generics that are in scope of the
 definition. For example, `tcx.type_of(def_id_of_my_struct)` would return the “self-view” of
 `MyStruct`: `Adt(Foo, &[Param(0), Param(1)])`.

How do we actually do the substitutions? There is a function for that too! You
use [`instantiate`] to replace a `GenericArgsRef` with  another list of types.

[Here is an example of actually using `instantiate` in the compiler][instantiatex].
The exact details are not too important, but in this piece of code, we happen to be
converting from the `rustc_hir::Ty` to a real `ty::Ty`. You can see that we first get some args
(`args`).  Then we call `type_of` to get a type and call `ty.instantiate(tcx, args)` to get a new
version of `ty` with the args made.

[`instantiate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/generic_args/struct.EarlyBinder.html#method.instantiate
[instantiatex]: https://github.com/rust-lang/rust/blob/8a562f9671e36cf29c9c794c2646bcf252d55535/compiler/rustc_hir_analysis/src/astconv/mod.rs#L905-L927

**Note on indices:** It is possible for the indices in `Param` to not match with what we expect. For
example, the index could be out of bounds or it could be the index of a lifetime when we were
expecting a type. These sorts of errors would be caught earlier in the compiler when translating
from a `rustc_hir::Ty` to a `ty::Ty`. If they occur later, that is a compiler bug.
