---
tags: rustc, ty
---

# The `ty` module: representing types

The `ty` module defines how the Rust compiler represents types internally. It also defines the
*typing context* (`tcx` or `TyCtxt`), which is the central data structure in the compiler.

## `ty::Ty`

When we talk about how rustc represents types,  we usually refer to a type called `Ty` . There are
quite a few modules and types for `Ty` in the compiler ([Ty documentation][ty]).

[ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/index.html

The specific `Ty` we are referring to is [`rustc::ty::Ty`][ty_ty] (and not
[`rustc_hir::Ty`][hir_ty]). The distinction is important, so we will discuss it first before going
into the details of `ty::Ty`.

[ty_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/type.Ty.html
[hir_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.Ty.html

## `rustc_hir::Ty` vs `ty::Ty`

The HIR in rustc can be thought of as the high-level intermediate representation. It is more or less
the AST (see [this chapter](hir.md)) as it represents the
syntax that the user wrote, and is obtained after parsing and some *desugaring*. It has a
representation of types, but in reality it reflects more of what the user wrote, that is, what they
wrote so as to represent that type.

In contrast, `ty::Ty` represents the semantics of a type, that is, the *meaning* of what the user
wrote. For example, `rustc_hir::Ty` would record the fact that a user used the name `u32` twice
in their program, but the `ty::Ty` would record the fact that both usages refer to the same type.

**Example: `fn foo(x: u32) → u32 { }`** In this function we see that `u32` appears twice. We know
that that is the same type, i.e. the function takes an argument and returns an argument of the same
type, but from the point of view of the HIR there would be two distinct type instances because these
are occurring in two different places in the program. That is, they have two
different [`Span`s][span] (locations).

[span]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html

**Example: `fn foo(x: &u32) -> &u32)`** In addition, HIR might have information left out. This type
`&u32` is incomplete, since in the full rust type there is actually a lifetime, but we didn’t need
to write those lifetimes. There are also some elision rules that insert information. The result may
look like  `fn foo<'a>(x: &'a u32) -> &'a u32)`.

In the HIR level, these things are not spelled out and you can say the picture is rather incomplete.
However, at the `ty::Ty` level, these details are added and it is complete. Moreover, we will have
exactly one `ty::Ty` for a given type, like `u32`, and that `ty::Ty` is used for all `u32`s in the
whole program, not a specific usage, unlike `rustc_hir::Ty`.

Here is a summary:

| [`rustc_hir::Ty`][hir_ty] | [`ty::Ty`][ty_ty] |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Describe the *syntax* of a type: what the user wrote (with some desugaring).  | Describe the *semantics* of a type: the meaning of what the user wrote. |
| Each `rustc_hir::Ty` has its own spans corresponding to the appropriate place in the program. | Doesn’t correspond to a single place in the user’s program. |
| `rustc_hir::Ty` has generics and lifetimes; however, some of those lifetimes are special markers like [`LifetimeName::Implicit`][implicit]. | `ty::Ty` has the full type, including generics and lifetimes, even if the user left them out |
| `fn foo(x: u32) → u32 { }` - Two `rustc_hir::Ty` representing each usage of `u32`. Each has its own `Span`s, etc.- `rustc_hir::Ty` doesn’t tell us that both are the same type | `fn foo(x: u32) → u32 { }` - One `ty::Ty` for all instances of `u32` throughout the program.- `ty::Ty` tells us that both usages of `u32` mean the same type. |
| `fn foo(x: &u32) -> &u32)`- Two `rustc_hir::Ty` again.- Lifetimes for the references show up in the `rustc_hir::Ty`s using a special marker, [`LifetimeName::Implicit`][implicit]. | `fn foo(x: &u32) -> &u32)`- A single `ty::Ty`.- The `ty::Ty` has the hidden lifetime param |

[implicit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/enum.LifetimeName.html#variant.Implicit

**Order** HIR is built directly from the AST, so it happens before any `ty::Ty` is produced. After
HIR is built, some basic type inference and type checking is done. During the type inference, we
figure out what the `ty::Ty` of everything is and we also check if the type of something is
ambiguous. The `ty::Ty` then, is used for type checking while making sure everything has the
expected type. The [`astconv` module][astconv], is where the code responsible for converting a
`rustc_hir::Ty` into a `ty::Ty` is located. This occurs during the type-checking phase,
but also in other parts of the compiler that want to ask questions like "what argument types does
this function expect"?.

[astconv]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_typeck/astconv/index.html

**How semantics drive the two instances of `Ty`** You can think of HIR as the perspective
of the type information that assumes the least. We assume two things are distinct until they are
proven to be the same thing. In other words, we know less about them, so we should assume less about
them.

They are syntactically two strings: `"u32"` at line N column 20 and `"u32"` at line N column 35. We
don’t know that they are the same yet. So, in the HIR we treat them as if they are different. Later,
we determine that they semantically are the same type and that’s the `ty::Ty` we use.

Consider another example: `fn foo<T>(x: T) -> u32` and suppose that someone invokes `foo::<u32>(0)`.
This means that `T` and `u32` (in this invocation) actually turns out to be the same type, so we
would eventually end up with the same `ty::Ty` in the end, but we have distinct `rustc_hir::Ty`.
(This is a bit over-simplified, though, since during type checking, we would check the function
generically and would still have a `T` distinct from `u32`. Later, when doing code generation,
we would always be handling "monomorphized" (fully substituted) versions of each function,
and hence we would know what `T` represents (and specifically that it is `u32`).

Here is one more example:

```rust
mod a {
    type X = u32;
    pub fn foo(x: X) -> i32 { 22 }
}
mod b {
    type X = i32;
    pub fn foo(x: X) -> i32 { x }
}
```

Here the type `X` will vary depending on context, clearly. If you look at the `rustc_hir::Ty`,
you will get back that `X` is an alias in both cases (though it will be mapped via name resolution
to distinct aliases). But if you look at the `ty::Ty` signature, it will be either `fn(u32) -> u32`
or `fn(i32) -> i32` (with type aliases fully expanded).

## `ty::Ty` implementation

[`rustc::ty::Ty`][ty_ty] is actually a type alias to [`&TyS`][tys] (more about that later). `TyS`
(Type Structure) is where the main functionality is located. You can ignore `TyS` struct in general;
you will basically never access it explicitly. We always pass it by reference using the `Ty` alias.
The only exception is to define inherent methods on types. In particular, `TyS` has a [`kind`][kind]
field of type [`TyKind`][tykind], which represents the key type information. `TyKind` is a big enum
which represents different kinds of types (e.g. primitives, references, abstract data types,
generics, lifetimes, etc). `TyS` also has 2 more fields, `flags` and `outer_exclusive_binder`. They
are convenient hacks for efficiency and summarize information about the type that we may want to
know, but they don’t come into the picture as much here.

[tys]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyS.html
[kind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyS.html#structfield.kind
[tykind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html

Note: `TyKind` is **NOT** the functional programming concept of *Kind*.

Whenever working with a `Ty` in the compiler, it is common to match on the kind of type:

```rust,ignore
fn foo(x: Ty<'tcx>) {
  match x.kind {
    ...
  }
}
```

The `kind` field is of type `TyKind<'tcx>`, which is an enum defining all of the different kinds of
types in the compiler.

> N.B. inspecting the `kind` field on types during type inference can be risky, as there may be
> inference variables and other things to consider, or sometimes types are not yet known and will
> become known later.

There are a lot of related types, and we’ll cover them in time (e.g regions/lifetimes,
“substitutions”, etc).

## `ty::TyKind` Variants

There are a bunch of variants on the `TyKind` enum, which you can see by looking at the rustdocs.
Here is a sampling:

[**Algebraic Data Types (ADTs)**]() An [*algebraic Data Type*][wikiadt] is a  `struct`, `enum` or
`union`.  Under the hood, `struct`, `enum` and `union` are actually implemented the same way: they
are both [`ty::TyKind::Adt`][kindadt].  It’s basically a user defined type. We will talk more about
these later.

[**Foreign**][kindforeign] Corresponds to `extern type T`.

[**Str**][kindstr] Is the type str. When the user writes `&str`, `Str` is the how we represent the
`str` part of that type.

[**Slice**][kindslice] Corresponds to `[T]`.

[**Array**][kindarray] Corresponds to `[T; n]`.

[**RawPtr**][kindrawptr] Corresponds to `*mut T` or `*const T`

[**Ref**][kindref] `Ref` stands for safe references, `&'a mut T` or `&'a T`. `Ref` has some
associated parts, like `Ty<'tcx>` which is the type that the reference references, `Region<'tcx>` is
the lifetime or region of the reference and `Mutability` if the reference is mutable or not.

[**Param**][kindparam] Represents a type parameter (e.g. the `T` in `Vec<T>`).

[**Error**][kinderr] Represents a type error somewhere so that we can print better diagnostics. We
will discuss this more later.

[**And Many More**...][kindvars]

[wikiadt]: https://en.wikipedia.org/wiki/Algebraic_data_type
[kindadt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Adt
[kindforeign]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Foreign
[kindstr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Str
[kindslice]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Slice
[kindarray]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Array
[kindrawptr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.RawPtr
[kindref]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Ref
[kindparam]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Param
[kinderr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variant.Error
[kindvars]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.TyKind.html#variants

## Interning

We create a LOT of types during compilation. For performance reasons, we allocate them from a global
memory pool, they are each allocated once from a long-lived *arena*. This is called _arena
allocation_. This system reduces allocations/deallocations of memory. It also allows for easy
comparison of types for equality: we implemented [`PartialEq for TyS`][peqimpl], so we can just
compare pointers. The [`CtxtInterners`] type contains a bunch of maps of interned types and the
arena itself.

[peqimpl]: https://github.com/rust-lang/rust/blob/3ee936378662bd2e74be951d6a7011a95a6bd84d/src/librustc/ty/mod.rs#L528-L534
[`CtxtInterners`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.CtxtInterners.html#structfield.arena

Each time we want to construct a type, the compiler doesn’t naively allocate from the buffer.
Instead, we check if that type was already constructed. If it was, we just get the same pointer we
had before, otherwise we make a fresh pointer. With this schema if we want to know if two types are
the same, all we need to do is compare the pointers which is efficient. `TyS` which represents types
is carefully setup so you never construct them on the stack. You always allocate them from this
arena and you always intern them so they are unique.

At the beginning of the compilation we make a buffer and each time we need to allocate a type we use
some of this memory buffer. If we run out of space we get another one. The lifetime of that buffer
is `'tcx`. Our types are tied to that lifetime, so when compilation finishes all the memory related
to that buffer is freed and our `'tcx` references would be invalid.


## The tcx and how it uses lifetimes

The `tcx` ("typing context") is the central data structure in the compiler. It is the context that
you use to perform all manner of queries. The struct `TyCtxt` defines a reference to this shared
context:

```rust,ignore
tcx: TyCtxt<'tcx>
//          ----
//          |
//          arena lifetime
```

As you can see, the `TyCtxt` type takes a lifetime parameter. When you see a reference with a
lifetime like `'tcx`, you know that it refers to arena-allocated data (or data that lives as long as
the arenas, anyhow).

## Allocating and working with types

To allocate a new type, you can use the various `mk_` methods defined on the `tcx`. These have names
that correspond mostly to the various kinds of types. For example:

```rust,ignore
let array_ty = tcx.mk_array(elem_ty, len * 2);
```

These methods all return a `Ty<'tcx>` – note that the lifetime you get back is the lifetime of the
arena that this `tcx` has access to. Types are always canonicalized and interned (so we never
allocate exactly the same type twice).

> NB. Because types are interned, it is possible to compare them for equality efficiently using `==`
> – however, this is almost never what you want to do unless you happen to be hashing and looking
> for duplicates. This is because often in Rust there are multiple ways to represent the same type,
> particularly once inference is involved. If you are going to be testing for type equality, you
> probably need to start looking into the inference code to do it right.

You can also find various common types in the `tcx` itself by accessing `tcx.types.bool`,
`tcx.types.char`, etc (see [`CommonTypes`] for more).

[`CommonTypes`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/context/struct.CommonTypes.html

## Beyond types: other kinds of arena-allocated data structures

In addition to types, there are a number of other arena-allocated data structures that you can
allocate, and which are found in this module. Here are a few examples:

- [`Substs`][subst], allocated with `mk_substs` – this will intern a slice of types, often used to
  specify the values to be substituted for generics (e.g. `HashMap<i32, u32>` would be represented
  as a slice `&'tcx [tcx.types.i32, tcx.types.u32]`).
- [`TraitRef`], typically passed by value – a **trait reference** consists of a reference to a trait
  along with its various type parameters (including `Self`), like `i32: Display` (here, the def-id
  would reference the `Display` trait, and the substs would contain `i32`). Note that `def-id` is
  defined and discussed in depth in the `AdtDef and DefId` section.
- [`Predicate`] defines something the trait system has to prove (see `traits` module).

[subst]: ./generic_arguments.html#subst
[`TraitRef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TraitRef.html
[`Predicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/enum.Predicate.html

## Import conventions

Although there is no hard and fast rule, the `ty` module tends to be used like so:

```rust,ignore
use ty::{self, Ty, TyCtxt};
```

In particular, since they are so common, the `Ty` and `TyCtxt` types are imported directly. Other
types are often referenced with an explicit `ty::` prefix (e.g. `ty::TraitRef<'tcx>`). But some
modules choose to import a larger or smaller set of names explicitly.

## ADTs Representation

Let's consider the example of a type like `MyStruct<u32>`, where `MyStruct` is defined like so:

```rust,ignore
struct MyStruct<T> { x: u32, y: T }
```

The type `MyStruct<u32>` would be an instance of `TyKind::Adt`:

```rust,ignore
Adt(&'tcx AdtDef, SubstsRef<'tcx>)
//  ------------  ---------------
//  (1)            (2)
//
// (1) represents the `MyStruct` part
// (2) represents the `<u32>`, or "substitutions" / generic arguments
```

There are two parts:

- The [`AdtDef`][adtdef] references the struct/enum/union but without the values for its type
  parameters. In our example, this is the `MyStruct` part *without* the argument `u32`.
    - Note that in the HIR, structs, enums and unions are represented differently, but in `ty::Ty`,
      they are all represented using `TyKind::Adt`.
- The [`SubstsRef`][substsref] is an interned list of values that are to be substituted for the
  generic parameters.  In our example of `MyStruct<u32>`, we would end up with a list like `[u32]`.
  We’ll dig more into generics and substitutions in a little bit.

[adtdef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.AdtDef.html
[substsref]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/subst/type.SubstsRef.html

**`AdtDef` and `DefId`**

For every type defined in the source code, there is a unique `DefId` (see [this
chapter](hir.md#identifiers-in-the-hir)). This includes ADTs and generics. In the `MyStruct<T>`
definition we gave above, there are two `DefId`s: one for `MyStruct` and one for `T`.  Notice that
the code above does not generate a new `DefId` for `u32` because it is not defined in that code (it
is only referenced).

`AdtDef` is more or less a wrapper around `DefId` with lots of useful helper methods. There is
essentially a one-to-one relationship between `AdtDef` and `DefId`. You can get the `AdtDef` for a
`DefId` with the [`tcx.adt_def(def_id)` query][adtdefq].  The `AdtDef`s are all interned (as you can
see `'tcx` lifetime on it).

[adtdefq]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.TyCtxt.html#method.adt_def


### Generics and substitutions

Given a generic type `MyType<A, B, …>`, we may want to swap out the generics `A, B, …` for some
other types (possibly other generics or concrete types). We do this a lot while doing type
inference, type checking, and trait solving. Conceptually, during these routines, we may find out
that one type is equal to another type and want to swap one out for the other and then swap that out
for another type and so on until we eventually get some concrete types (or an error).

In rustc this is done using the `SubstsRef` that we mentioned above (“substs” = “substitutions”).
Conceptually, you can think of `SubstsRef` of a list of types that are to be substituted for the
generic type parameters of the ADT.

`SubstsRef` is a type alias of `List<GenericArg<'tcx>>` (see [`List` rustdocs][list]).
[`GenericArg`] is essentially a space-efficient wrapper around [`GenericArgKind`], which is an enum
indicating what kind of generic the type parameter is (type, lifetime, or const).  Thus, `SubstsRef`
is conceptually like a `&'tcx [GenericArgKind<'tcx>]` slice (but it is actually a `List`).

[list]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.List.html
[`GenericArg`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/subst/struct.GenericArg.html
[`GenericArgKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/subst/enum.GenericArgKind.html

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
- There would be a `SubstsRef` containing the list `[GenericArgKind::Type(Ty(T))]`
    - The `Ty(T)` here is my shorthand for entire other `ty::Ty` that has `TyKind::Param`, which we
      mentioned in the previous point.
- This is one `TyKind::Adt` containing the `AdtDef` of `MyStruct` with the `SubstsRef` above.

Finally, we will quickly mention the
[`Generics`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/struct.Generics.html) type. It
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
the index `0` and use it to find the right type substitution: looking at `Foo`'s `SubstsRef`, we
have the list `[u32, f32]` , since we want to replace index `0`, we take the 0-th index of this
list, which is `u32`. Voila!

You may have a couple of followup questions…

 **`type_of`** How do we get the “generic type of `x`"? You can get the type of pretty much anything
 with the   `tcx.type_of(def_id)` query. In this case, we would pass the `DefId` of the field `x`.
 The `type_of` query always returns the definition with the generics that are in scope of the
 definition. For example, `tcx.type_of(def_id_of_my_struct)` would return the “self-view” of
 `MyStruct`: `Adt(Foo, &[Param(0), Param(1)])`.

**`subst`** How do we actually do the substitutions? There is a function for that too! You use
[`subst`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/subst/trait.Subst.html) to
replace a `SubstRef` with another list of types.

[Here is an example of actually using `subst` in the compiler][substex].  The exact details are not
too important, but in this piece of code, we happen to be converting from the `rustc_hir::Ty` to
a real `ty::Ty`. You can see that we first get some substitutions (`substs`).  Then we call
`type_of` to get a type and call `ty.subst(substs)` to get a new version of `ty` with
the substitutions made.

[substex]: https://github.com/rust-lang/rust/blob/597f432489f12a3f33419daa039ccef11a12c4fd/src/librustc_typeck/astconv.rs#L942-L953

**Note on indices:** It is possible for the indices in `Param` to not match with what we expect. For
example, the index could be out of bounds or it could be the index of a lifetime when we were
expecting a type. These sorts of errors would be caught earlier in the compiler when translating
from a `rustc_hir::Ty` to a `ty::Ty`. If they occur later, that is a compiler bug.

### `TypeFoldable` and `TypeFolder`

How is this `subst` query actually implemented? As you can imagine, we might want to do
substitutions on a lot of different things. For example, we might want to do a substitution directly
on a type like we did with `Vec` above. But we might also have a more complex type with other types
nested inside that also need substitutions.

The answer is a couple of traits:
[`TypeFoldable`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/fold/trait.TypeFoldable.html)
and
[`TypeFolder`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/fold/trait.TypeFolder.html).

- `TypeFoldable` is implemented by types that embed type information. It allows you to recursively
  process the contents of the `TypeFoldable` and do stuff to them.
- `TypeFolder` defines what you want to do with the types you encounter while processing the
  `TypeFoldable`.

For example, the `TypeFolder` trait has a method
[`fold_ty`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/fold/trait.TypeFolder.html#method.fold_ty)
that takes a type as input a type and returns a new type as a result. `TypeFoldable` invokes the
`TypeFolder` `fold_foo` methods on itself, giving the `TypeFolder` access to its contents (the
types, regions, etc that are contained within).

You can think of it with this analogy to the iterator combinators we have come to love in rust:

```rust,ignore
vec.iter().map(|e1| foo(e2)).collect()
//             ^^^^^^^^^^^^ analogous to `TypeFolder`
//         ^^^ analogous to `Typefoldable`
```

So to reiterate:

- `TypeFolder`  is a trait that defines a “map” operation.
- `TypeFoldable`  is a trait that is implemented by things that embed types.

In the case of `subst`, we can see that it is implemented as a `TypeFolder`:
[`SubstFolder`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/subst/struct.SubstFolder.html).
Looking at its implementation, we see where the actual substitutions are happening.

However, you might also notice that the implementation calls this `super_fold_with` method. What is
that? It is a method of `TypeFoldable`. Consider the following `TypeFoldable` type `MyFoldable`:

```rust,ignore
struct MyFoldable<'tcx> {
  def_id: DefId,
  ty: Ty<'tcx>,
}
```

The `TypeFolder` can call `super_fold_with` on `MyFoldable` if it just wants to replace some of the
fields of `MyFoldable` with new values. If it instead wants to replace the whole `MyFoldable` with a
different one, it would call `fold_with` instead (a different method on `TypeFoldable`).

In almost all cases, we don’t want to replace the whole struct; we only want to replace `ty::Ty`s in
the struct, so usually we call `super_fold_with`. A typical implementation that `MyFoldable` could
have might do something like this:

```rust,ignore
my_foldable: MyFoldable<'tcx>
my_foldable.subst(..., subst)

impl TypeFoldable for MyFoldable {
  fn super_fold_with(&self, folder: &mut impl TypeFolder<'tcx>) -> MyFoldable {
    MyFoldable {
      def_id: self.def_id.fold_with(folder),
      ty: self.ty.fold_with(folder),
    }
  }

  fn super_visit_with(..) { }
}
```

Notice that here, we implement `super_fold_with` to go over the fields of `MyFoldable` and call
`fold_with` on *them*. That is, a folder may replace  `def_id` and `ty`, but not the whole
`MyFoldable` struct.

Here is another example to put things together: suppose we have a type like `Vec<Vec<X>>`. The
`ty::Ty` would look like: `Adt(Vec, &[Adt(Vec, &[Param(X)])])`. If we want to do `subst(X => u32)`,
then we would first look at the overall type. We would see that there are no substitutions to be
made at the outer level, so we would descend one level and look at `Adt(Vec, &[Param(X)])`. There
are still no substitutions to be made here, so we would descend again. Now we are looking at
`Param(X)`, which can be substituted, so we replace it with `u32`. We can’t descend any more, so we
are done, and  the overall result is `Adt(Vec, &[Adt(Vec, &[u32])])`.

One last thing to mention: often when folding over a `TypeFoldable`, we don’t want to change most
things. We only want to do something when we reach a type. That means there may be a lot of
`TypeFoldable` types whose implementations basically just forward to their fields’ `TypeFoldable`
implementations. Such implementations of `TypeFoldable` tend to be pretty tedious to write by hand.
For this reason, there is a `derive` macro that allows you to `#![derive(TypeFoldable)]`. It is
defined
[here](https://github.com/rust-lang/rust/blob/master/src/librustc_macros/src/type_foldable.rs).

**`subst`** In the case of substitutions the [actual
folder](https://github.com/rust-lang/rust/blob/04e69e4f4234beb4f12cc76dcc53e2cc4247a9be/src/librustc/ty/subst.rs#L467-L482)
is going to be doing the indexing we’ve already mentioned. There we define a `Folder` and call
`fold_with` on the `TypeFoldable` to process yourself.  Then
[fold_ty](https://github.com/rust-lang/rust/blob/04e69e4f4234beb4f12cc76dcc53e2cc4247a9be/src/librustc/ty/subst.rs#L545-L573)
the method that process each type it looks for a `ty::Param` and for those it replaces it for
something from the list of substitutions, otherwise recursively process the type.  To replace it,
calls
[ty_for_param](https://github.com/rust-lang/rust/blob/04e69e4f4234beb4f12cc76dcc53e2cc4247a9be/src/librustc/ty/subst.rs#L589-L624)
and all that does is index into the list of substitutions with the index of the `Param`.

## Type errors

There is a `TyKind::Error` that is produced when the user makes a type error. The idea is that
we would propagate this type and suppress other errors that come up due to it so as not to overwhelm
the user with cascading compiler error messages.

There is an **important invariant** for `TyKind::Error`. You should **never** return the 'error
type' unless you **know** that an error has already been reported to the user. This is usually
because (a) you just reported it right there or (b) you are propagating an existing Error type (in
which case the error should've been reported when that error type was produced).

It's important to maintain this invariant because the whole point of the `Error` type is to suppress
other errors -- i.e., we don't report them. If we were to produce an `Error` type without actually
emitting an error to the user, then this could cause later errors to be suppressed, and the
compilation might inadvertently succeed!

Sometimes there is a third case. You believe that an error has been reported, but you believe it
would've been reported earlier in the compilation, not locally. In that case, you can invoke
[`delay_span_bug`] This will make a note that you expect compilation to yield an error -- if however
compilation should succeed, then it will trigger a compiler bug report.

[`delay_span_bug`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html#method.delay_span_bug

## Question: Why not substitute “inside” the `AdtDef`?

Recall that we represent a generic struct with `(AdtDef, substs)`. So why bother with this scheme?

Well, the alternate way we could have choosen to represent types would be to always create a new,
fully-substituted form of the `AdtDef` where all the types are already substituted. This seems like
less of a hassle. However, the `(AdtDef, substs)` scheme has some advantages over this.

First, `(AdtDef, substs)` scheme has an efficiency win:

```rust,ignore
struct MyStruct<T> {
  ... 100s of fields ...
}

// Want to do: MyStruct<A> ==> MyStruct<B>
```

in an example like this, we can subst from `MyStruct<A>` to `MyStruct<B>` (and so on) very cheaply,
by just replacing the one reference to `A` with `B`. But if we eagerly substituted all the fields,
that could be a lot more work because we might have to go through all of the fields in the `AdtDef`
and update all of their types.

A bit more deeply, this corresponds to structs in Rust being [*nominal* types][nominal] — which
means that they are defined by their *name* (and that their contents are then indexed from the
definition of that name, and not carried along “within” the type itself).

[nominal]: https://en.wikipedia.org/wiki/Nominal_type_system
