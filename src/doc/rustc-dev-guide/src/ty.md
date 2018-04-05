# The `ty` module: representing types

The `ty` module defines how the Rust compiler represents types
internally. It also defines the *typing context* (`tcx` or `TyCtxt`),
which is the central data structure in the compiler.

## The tcx and how it uses lifetimes

The `tcx` ("typing context") is the central data structure in the
compiler. It is the context that you use to perform all manner of
queries. The struct `TyCtxt` defines a reference to this shared context:

```rust,ignore
tcx: TyCtxt<'a, 'gcx, 'tcx>
//          --  ----  ----
//          |   |     |
//          |   |     innermost arena lifetime (if any)
//          |   "global arena" lifetime
//          lifetime of this reference
```

As you can see, the `TyCtxt` type takes three lifetime parameters.
These lifetimes are perhaps the most complex thing to understand about
the tcx. During Rust compilation, we allocate most of our memory in
**arenas**, which are basically pools of memory that get freed all at
once. When you see a reference with a lifetime like `'tcx` or `'gcx`,
you know that it refers to arena-allocated data (or data that lives as
long as the arenas, anyhow).

We use two distinct levels of arenas. The outer level is the "global
arena". This arena lasts for the entire compilation: so anything you
allocate in there is only freed once compilation is basically over
(actually, when we shift to executing LLVM).

To reduce peak memory usage, when we do type inference, we also use an
inner level of arena. These arenas get thrown away once type inference
is over. This is done because type inference generates a lot of
"throw-away" types that are not particularly interesting after type
inference completes, so keeping around those allocations would be
wasteful.

Often, we wish to write code that explicitly asserts that it is not
taking place during inference. In that case, there is no "local"
arena, and all the types that you can access are allocated in the
global arena.  To express this, the idea is to use the same lifetime
for the `'gcx` and `'tcx` parameters of `TyCtxt`. Just to be a touch
confusing, we tend to use the name `'tcx` in such contexts. Here is an
example:

```rust,ignore
fn not_in_inference<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    //                                        ----  ----
    //                                        Using the same lifetime here asserts
    //                                        that the innermost arena accessible through
    //                                        this reference *is* the global arena.
}
```

In contrast, if we want to code that can be usable during type inference, then
you need to declare a distinct `'gcx` and `'tcx` lifetime parameter:

```rust,ignore
fn maybe_in_inference<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>, def_id: DefId) {
    //                                                ----  ----
    //                                        Using different lifetimes here means that
    //                                        the innermost arena *may* be distinct
    //                                        from the global arena (but doesn't have to be).
}
```

### Allocating and working with types

Rust types are represented using the `Ty<'tcx>` defined in the `ty`
module (not to be confused with the `Ty` struct from [the HIR]). This
is in fact a simple type alias for a reference with `'tcx` lifetime:

```rust,ignore
pub type Ty<'tcx> = &'tcx TyS<'tcx>;
```

[the HIR]: ./hir.html

You can basically ignore the `TyS` struct – you will basically never
access it explicitly. We always pass it by reference using the
`Ty<'tcx>` alias – the only exception I think is to define inherent
methods on types. Instances of `TyS` are only ever allocated in one of
the rustc arenas (never e.g. on the stack).

One common operation on types is to **match** and see what kinds of
types they are. This is done by doing `match ty.sty`, sort of like this:

```rust,ignore
fn test_type<'tcx>(ty: Ty<'tcx>) {
    match ty.sty {
        ty::TyArray(elem_ty, len) => { ... }
        ...
    }
}
```

The `sty` field (the origin of this name is unclear to me; perhaps
structural type?) is of type `TyKind<'tcx>`, which is an enum
defining all of the different kinds of types in the compiler.

> N.B. inspecting the `sty` field on types during type inference can be
> risky, as there may be inference variables and other things to
> consider, or sometimes types are not yet known that will become
> known later.).

To allocate a new type, you can use the various `mk_` methods defined
on the `tcx`. These have names that correpond mostly to the various kinds
of type variants. For example:

```rust,ignore
let array_ty = tcx.mk_array(elem_ty, len * 2);
```

These methods all return a `Ty<'tcx>` – note that the lifetime you
get back is the lifetime of the innermost arena that this `tcx` has
access to. In fact, types are always canonicalized and interned (so we
never allocate exactly the same type twice) and are always allocated
in the outermost arena where they can be (so, if they do not contain
any inference variables or other "temporary" types, they will be
allocated in the global arena). However, the lifetime `'tcx` is always
a safe approximation, so that is what you get back.

> NB. Because types are interned, it is possible to compare them for
> equality efficiently using `==` – however, this is almost never what
> you want to do unless you happen to be hashing and looking for
> duplicates. This is because often in Rust there are multiple ways to
> represent the same type, particularly once inference is involved. If
> you are going to be testing for type equality, you probably need to
> start looking into the inference code to do it right.

You can also find various common types in the `tcx` itself by accessing
`tcx.types.bool`, `tcx.types.char`, etc (see `CommonTypes` for more).

### Beyond types: other kinds of arena-allocated data structures

In addition to types, there are a number of other arena-allocated data
structures that you can allocate, and which are found in this
module. Here are a few examples:

- [`Substs`][subst], allocated with `mk_substs` – this will intern a slice of
  types, often used to specify the values to be substituted for generics
  (e.g. `HashMap<i32, u32>` would be represented as a slice
  `&'tcx [tcx.types.i32, tcx.types.u32]`).
- `TraitRef`, typically passed by value – a **trait reference**
  consists of a reference to a trait along with its various type
  parameters (including `Self`), like `i32: Display` (here, the def-id
  would reference the `Display` trait, and the substs would contain
  `i32`).
- `Predicate` defines something the trait system has to prove (see `traits`
  module).

[subst]: ./kinds.html#subst

### Import conventions

Although there is no hard and fast rule, the `ty` module tends to be used like
so:

```rust,ignore
use ty::{self, Ty, TyCtxt};
```

In particular, since they are so common, the `Ty` and `TyCtxt` types
are imported directly. Other types are often referenced with an
explicit `ty::` prefix (e.g. `ty::TraitRef<'tcx>`). But some modules
choose to import a larger or smaller set of names explicitly.
