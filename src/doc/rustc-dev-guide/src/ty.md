# The `ty` module: representing types

<!-- toc -->

The `ty` module defines how the Rust compiler represents types internally. It also defines the
*typing context* (`tcx` or `TyCtxt`), which is the central data structure in the compiler.

## `ty::Ty`

When we talk about how rustc represents types,  we usually refer to a type called `Ty` . There are
quite a few modules and types for `Ty` in the compiler ([Ty documentation][ty]).

[ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/index.html

The specific `Ty` we are referring to is [`rustc_middle::ty::Ty`][ty_ty] (and not
[`rustc_hir::Ty`][hir_ty]). The distinction is important, so we will discuss it first before going
into the details of `ty::Ty`.

[ty_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html
[hir_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.Ty.html

## `rustc_hir::Ty` vs `ty::Ty`

The HIR in rustc can be thought of as the high-level intermediate representation. It is more or less
the AST (see [this chapter](hir.md)) as it represents the
syntax that the user wrote, and is obtained after parsing and some *desugaring*. It has a
representation of types, but in reality it reflects more of what the user wrote, that is, what they
wrote so as to represent that type.

In contrast, `ty::Ty` represents the semantics of a type, that is, the *meaning* of what the user
wrote. For example, `rustc_hir::Ty` would record the fact that a user used the name `u32` twice
in their program, but the `ty::Ty` would record the fact that both usages refer to the same type.

**Example: `fn foo(x: u32) → u32 { x }`**

In this function, we see that `u32` appears twice. We know
that that is the same type,
i.e. the function takes an argument and returns an argument of the same type,
but from the point of view of the HIR,
there would be two distinct type instances because these
are occurring in two different places in the program.
That is, they have two different [`Span`s][span] (locations).

[span]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html

**Example: `fn foo(x: &u32) -> &u32`**

In addition, HIR might have information left out. This type
`&u32` is incomplete, since in the full Rust type there is actually a lifetime, but we didn’t need
to write those lifetimes. There are also some elision rules that insert information. The result may
look like  `fn foo<'a>(x: &'a u32) -> &'a u32`.

In the HIR level, these things are not spelled out and you can say the picture is rather incomplete.
However, at the `ty::Ty` level, these details are added and it is complete. Moreover, we will have
exactly one `ty::Ty` for a given type, like `u32`, and that `ty::Ty` is used for all `u32`s in the
whole program, not a specific usage, unlike `rustc_hir::Ty`.

Here is a summary:

| [`rustc_hir::Ty`][hir_ty] | [`ty::Ty`][ty_ty] |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Describe the *syntax* of a type: what the user wrote (with some desugaring).  | Describe the *semantics* of a type: the meaning of what the user wrote. |
| Each `rustc_hir::Ty` has its own spans corresponding to the appropriate place in the program. | Doesn’t correspond to a single place in the user’s program. |
| `rustc_hir::Ty` has generics and lifetimes; however, some of those lifetimes are special markers like [`LifetimeKind::Implicit`][implicit]. | `ty::Ty` has the full type, including generics and lifetimes, even if the user left them out |
| `fn foo(x: u32) -> u32 { }` - Two `rustc_hir::Ty` representing each usage of `u32`, each has its own `Span`s, and `rustc_hir::Ty` doesn’t tell us that both are the same type | `fn foo(x: u32) -> u32 { }` - One `ty::Ty` for all instances of `u32` throughout the program, and `ty::Ty` tells us that both usages of `u32` mean the same type. |
| `fn foo(x: &u32) -> &u32 { }` - Two `rustc_hir::Ty` again. Lifetimes for the references show up in the `rustc_hir::Ty`s using a special marker, [`LifetimeKind::Implicit`][implicit]. | `fn foo(x: &u32) -> &u32 { }`- A single `ty::Ty`. The `ty::Ty` has the hidden lifetime param. |

[implicit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.LifetimeKind.html#variant.Implicit

**Order**

HIR is built directly from the AST, so it happens before any `ty::Ty` is produced. After
HIR is built, some basic type inference and type checking is done. During the type inference, we
figure out what the `ty::Ty` of everything is and we also check if the type of something is
ambiguous. The `ty::Ty` is then used for type checking while making sure everything has the
expected type. The [`hir_ty_lowering` module][hir_ty_lowering] is where the code responsible for
lowering a `rustc_hir::Ty` to a `ty::Ty` is located. The main routine used is `lower_ty`.
This occurs during the type-checking phase, but also in other parts of the compiler that want to ask
questions like "what argument types does this function expect?"

[hir_ty_lowering]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/hir_ty_lowering/index.html

**How semantics drive the two instances of `Ty`**

You can think of HIR as the perspective
of the type information that assumes the least. We assume two things are distinct until they are
proven to be the same thing. In other words, we know less about them, so we should assume less about
them.

They are syntactically two strings: `"u32"` at line N column 20 and `"u32"` at line N column 35. We
don’t know that they are the same yet. So, in the HIR we treat them as if they are different. Later,
we determine that they semantically are the same type and that’s the `ty::Ty` we use.

Consider another example: `fn foo<T>(x: T) -> u32`. Suppose that someone invokes `foo::<u32>(0)`.
This means that `T` and `u32` (in this invocation) actually turns out to be the same type, so we
would eventually end up with the same `ty::Ty` in the end, but we have distinct `rustc_hir::Ty`.
(This is a bit over-simplified, though, since during type checking, we would check the function
generically and would still have a `T` distinct from `u32`. Later, when doing code generation,
we would always be handling "monomorphized" (fully substituted) versions of each function,
and hence we would know what `T` represents (and specifically that it is `u32`).)

Here is one more example:

```rust
mod a {
    type X = u32;
    pub fn foo(x: X) -> u32 { 22 }
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

[`rustc_middle::ty::Ty`][ty_ty] is actually a wrapper around
[`Interned<WithCachedTypeInfo<TyKind>>`][tykind].
You can ignore `Interned` in general; you will basically never access it explicitly.
We always hide them within `Ty` and skip over it via `Deref` impls or methods.
`TyKind` is a big enum
with variants to represent many different Rust types
(e.g. primitives, references, algebraic data types, generics, lifetimes, etc).
`WithCachedTypeInfo` has a few cached values like `flags` and `outer_exclusive_binder`. They
are convenient hacks for efficiency and summarize information about the type that we may want to
know, but they don’t come into the picture as much here. Finally, [`Interned`](./memory.md) allows
the `ty::Ty` to be a thin pointer-like
type. This allows us to do cheap comparisons for equality, along with the other
benefits of interning.

[tykind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html

## Allocating and working with types

To allocate a new type, you can use the various `new_*` methods defined on
[`Ty`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html).
These have names
that correspond mostly to the various kinds of types. For example:

```rust,ignore
let array_ty = Ty::new_array_with_const_len(tcx, ty, count);
```

These methods all return a `Ty<'tcx>` – note that the lifetime you get back is the lifetime of the
arena that this `tcx` has access to. Types are always canonicalized and interned (so we never
allocate exactly the same type twice).

You can also find various common types in the `tcx` itself by accessing its fields:
`tcx.types.bool`, `tcx.types.char`, etc. (See [`CommonTypes`] for more.)

[`CommonTypes`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.CommonTypes.html

<!-- N.B: This section is linked from the type comparison internal lint. -->
## Comparing types

Because types are interned, it is possible to compare them for equality efficiently using `==`
– however, this is almost never what you want to do unless you happen to be hashing and looking
for duplicates. This is because often in Rust there are multiple ways to represent the same type,
particularly once inference is involved.

For example, the type `{integer}` (`ty::Infer(ty::IntVar(..))` an integer inference variable,
the type of an integer literal like `0`) and `u8` (`ty::UInt(..)`) should often be treated as
equal when testing whether they can be assigned to each other (which is a common operation in
diagnostics code). `==` on them will return `false` though, since they are different types.

The simplest way to compare two types correctly requires an inference context (`infcx`).
If you have one, you can use `infcx.can_eq(param_env, ty1, ty2)`
to check whether the types can be made equal.
This is typically what you want to check during diagnostics, which is concerned with questions such
as whether two types can be assigned to each other, not whether they're represented identically in
the compiler's type-checking layer.

When working with an inference context, you have to be careful to ensure that potential inference
variables inside the types actually belong to that inference context. If you are in a function
that has access to an inference context already, this should be the case. Specifically, this is the
case during HIR type checking or MIR borrow checking.

Another consideration is normalization. Two types may actually be the same, but one is behind an
associated type. To compare them correctly, you have to normalize the types first. This is
primarily a concern during HIR type checking and with all types from a `TyCtxt` query
(for example from `tcx.type_of()`).

When a `FnCtxt` or an `ObligationCtxt` is available during type checking, `.normalize(ty)`
should be used on them to normalize the type. After type checking, diagnostics code can use
`tcx.normalize_erasing_regions(ty)`.

There are also cases where using `==` on `Ty` is fine. This is for example the case in late lints
or after monomorphization, since type checking has been completed, meaning all inference variables
are resolved and all regions have been erased. In these cases, if you know that inference variables
or normalization won't be a concern, `#[allow]` or `#[expect]`ing the lint is recommended.

When diagnostics code does not have access to an inference context, it should be threaded through
the function calls if one is available in some place (like during type checking).

If no inference context is available at all, then one can be created as described in
[type-inference]. But this is only useful when the involved types (for example, if
they came from a query like `tcx.type_of()`) are actually substituted with fresh
inference variables using [`fresh_args_for_item`]. This can be used to answer questions
like "can `Vec<T>` for any `T` be unified with `Vec<u32>`?".

[type-inference]: ./type-inference.md#creating-an-inference-context
[`fresh_args_for_item`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.fresh_substs_for_item

## `ty::TyKind` Variants

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

There are many variants on the `TyKind` enum, which you can see by looking at its
[documentation][tykind]. Here is a sampling:

- [**Algebraic Data Types (ADTs)**][kindadt] An [*algebraic data type*][wikiadt] is a  `struct`,
  `enum` or `union`.  Under the hood, `struct`, `enum` and `union` are actually implemented
  the same way: they are all [`ty::TyKind::Adt`][kindadt].  It’s basically a user defined type.
  We will talk more about these later.
- [**Foreign**][kindforeign] Corresponds to `extern type T`.
- [**Str**][kindstr] Is the type str. When the user writes `&str`, `Str` is the how we represent the
  `str` part of that type.
- [**Slice**][kindslice] Corresponds to `[T]`.
- [**Array**][kindarray] Corresponds to `[T; n]`.
- [**RawPtr**][kindrawptr] Corresponds to `*mut T` or `*const T`.
- [**Ref**][kindref] `Ref` stands for safe references, `&'a mut T` or `&'a T`. `Ref` has some
  associated parts, like `Ty<'tcx>` which is the type that the reference references.
  `Region<'tcx>` is the lifetime or region of the reference and `Mutability` if the reference
  is mutable or not.
- [**Param**][kindparam] Represents a type parameter (e.g. the `T` in `Vec<T>`).
- [**Error**][kinderr] Represents a type error somewhere so that we can print better diagnostics. We
  will discuss this more later.
- [**And many more**...][kindvars]

[wikiadt]: https://en.wikipedia.org/wiki/Algebraic_data_type
[kindadt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Adt
[kindforeign]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Foreign
[kindstr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Str
[kindslice]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Slice
[kindarray]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Array
[kindrawptr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.RawPtr
[kindref]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Ref
[kindparam]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Param
[kinderr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Error
[kindvars]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variants

## Import conventions

Although there is no hard and fast rule, the `ty` module tends to be used like so:

```rust,ignore
use ty::{self, Ty, TyCtxt};
```

In particular, since they are so common, the `Ty` and `TyCtxt` types are imported directly. Other
types are often referenced with an explicit `ty::` prefix (e.g. `ty::TraitRef<'tcx>`). But some
modules choose to import a larger or smaller set of names explicitly.

## Type errors

There is a `TyKind::Error` that is produced when the user makes a type error. The idea is that
we would propagate this type and suppress other errors that come up due to it so as not to overwhelm
the user with cascading compiler error messages.

There is an **important invariant** for `TyKind::Error`. The compiler should
**never** produce `Error` unless we **know** that an error has already been
reported to the user. This is usually
because (a) you just reported it right there or (b) you are propagating an existing Error type (in
which case the error should've been reported when that error type was produced).

It's important to maintain this invariant because the whole point of the `Error` type is to suppress
other errors -- i.e., we don't report them. If we were to produce an `Error` type without actually
emitting an error to the user, then this could cause later errors to be suppressed, and the
compilation might inadvertently succeed!

Sometimes there is a third case. You believe that an error has been reported, but you believe it
would've been reported earlier in the compilation, not locally. In that case, you can create a
"delayed bug" with [`delayed_bug`] or [`span_delayed_bug`]. This will make a note that you expect
compilation to yield an error -- if however compilation should succeed, then it will trigger a
compiler bug report.

[`delayed_bug`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxt.html#method.delayed_bug
[`span_delayed_bug`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxt.html#method.span_delayed_bug

For added safety, it's not actually possible to produce a `TyKind::Error` value
outside of [`rustc_middle::ty`][ty]; there is a private member of
`TyKind::Error` that prevents it from being constructable elsewhere. Instead,
one should use the [`Ty::new_error`][terr] or
[`Ty::new_error_with_message`][terrmsg] methods. These methods either take an `ErrorGuaranteed`
or call `span_delayed_bug` before returning an interned `Ty` of kind `Error`. If you
were already planning to use [`span_delayed_bug`], then you can just pass the
span and message to [`ty_error_with_message`][terrmsg] instead to avoid
a redundant delayed bug.

[terr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#method.new_error
[terrmsg]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#method.new_error_with_message


## `TyKind` variant shorthand syntax

When looking at the debug output of `Ty` or simply talking about different types in the compiler, you may encounter syntax that is not valid rust but is used to concisely represent internal information about types. Below is a quick reference cheat sheet to tell what the various syntax actually means, these should be covered in more depth in later chapters.

- Generic parameters: `{name}/#{index}` e.g. `T/#0`, where `index` corresponds to its position in the list of generic parameters
- Inference variables: `?{id}` e.g. `?x`/`?0`, where `id` identifies the inference variable
- Variables from binders: `^{binder}_{index}` e.g. `^0_x`/`^0_2`, where `binder` and `index` identify which variable from which binder is being referred to
- Placeholders: `!{id}` or `!{id}_{universe}` e.g. `!x`/`!0`/`!x_2`/`!0_2`, representing some unique type in the specified universe. The universe is often elided when it is `0`
