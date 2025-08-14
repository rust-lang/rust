# Invariants of the type system

FIXME: This file talks about invariants of the type system as a whole, not only the solver

There are a lot of invariants - things the type system guarantees to be true at all times -
which are desirable or expected from other languages and type systems. Unfortunately, quite
a few of them do not hold in Rust right now. This is either fundamental to its design or
caused by bugs, and something that may change in the future.

It is important to know about the things you can assume while working on, and with, the
type system, so here's an incomplete and unofficial list of invariants of
the core type system:

- ✅: this invariant mostly holds, with some weird exceptions or current bugs
- ❌: this invariant does not hold, and is unlikely to do so in the future; do not rely on it for soundness or have to be incredibly careful when doing so

### `wf(X)` implies `wf(normalize(X))` ✅

If a type containing aliases is well-formed, it should also be
well-formed after normalizing said aliases. We rely on this as
otherwise we would have to re-check for well-formedness for these
types.

This currently does not hold due to a type system unsoundness: [#84533](https://github.com/rust-lang/rust/issues/84533).

### Structural equality modulo regions implies semantic equality ✅

If you have a some type and equate it to itself after replacing any regions with unique
inference variables in both the lhs and rhs, the now potentially structurally different
types should still be equal to each other.

This is needed to prevent goals from succeeding in HIR typeck and then failing in MIR borrowck.
If this invariant is broken, MIR typeck ends up failing with an ICE.

### Applying inference results from a goal does not change its result ❌

TODO: this invariant is formulated in a weird way and needs to be elaborated.
Pretty much: I would like this check to only fail if there's a solver bug:
<https://github.com/rust-lang/rust/blob/2ffeb4636b4ae376f716dc4378a7efb37632dc2d/compiler/rustc_trait_selection/src/solve/eval_ctxt.rs#L391-L407>.
We should readd this check and see where it breaks :3

If we prove some goal/equate types/whatever, apply the resulting inference constraints,
and then redo the original action, the result should be the same.

This unfortunately does not hold - at least in the new solver - due to a few annoying reasons.

### The trait solver has to be *locally sound* ✅

This means that we must never return *success* for goals for which no `impl` exists. That would
mean we assume a trait is implemented even though it is not, which is very likely to result in
actual unsoundness. When using `where`-bounds to prove a goal, the `impl` will be provided by the
user of the item.

This invariant only holds if we check region constraints. As we do not check region constraints
during implicit negative overlap check in coherence, this invariant is broken there. As this check
relies on *completeness* of the trait solver, it is not able to use the current region constraints
check - `InferCtxt::resolve_regions` - as its handling of type outlives goals is incomplete.

### Normalization of semantically equal aliases in empty environments results in a unique type ✅

Normalization for alias types/consts has to have a unique result. Otherwise we can easily 
implement transmute in safe code. Given the following function, we have to make sure that
the input and output types always get normalized to the same concrete type.

```rust
fn foo<T: Trait>(
    x: <T as Trait>::Assoc
) -> <T as Trait>::Assoc {
    x
}
```

Many of the currently known unsound issues end up relying on this invariant being broken.
It is however very difficult to imagine a sound type system without this invariant, so
the issue is that the invariant is broken, not that we incorrectly rely on it.

### The type system is complete ❌

The type system is not complete.
It often adds unnecessary inference constraints, and errors even though the goal could hold.

- method selection
- opaque type inference
- handling type outlives constraints
- preferring `ParamEnv` candidates over `Impl` candidates during candidate selection
in the trait solver

### Goals keep their result from HIR typeck afterwards ✅

Having a goal which succeeds during HIR typeck but fails when being reevaluated during MIR borrowck causes ICE, e.g. [#140211](https://github.com/rust-lang/rust/issues/140211).

Having a goal which succeeds during HIR typeck but fails after being instantiated is unsound, e.g. [#140212](https://github.com/rust-lang/rust/issues/140212).

It is interesting that we allow some incompleteness in the trait solver while still maintaining this limitation. It would be nice if there was a clear way to separate the "allowed incompleteness" from behavior which would break this invariant.

#### Normalization must not change results

This invariant is relied on to allow the normalization of generic aliases. Breaking
it can easily result in unsoundness, e.g. [#57893](https://github.com/rust-lang/rust/issues/57893).

#### Goals may still overflow after instantiation

This happens they start to hit the recursion limit.
We also have diverging aliases which are scuffed.
It's unclear how these should be handled :3

### Trait goals in empty environments are proven by a unique impl ✅

If a trait goal holds with an empty environment, there should be a unique `impl`,
either user-defined or builtin, which is used to prove that goal. This is
necessary to select unique methods and associated items.

We do however break this invariant in a few cases, some of which are due to bugs, some by design:
- *marker traits* are allowed to overlap as they do not have associated items
- *specialization* allows specializing impls to overlap with their parent
- the builtin trait object trait implementation can overlap with a user-defined impl:
[#57893](https://github.com/rust-lang/rust/issues/57893)


#### The type system is complete during the implicit negative overlap check in coherence ✅

For more on overlap checking, see [Coherence chapter].

During the implicit negative overlap check in coherence,
we must never return *error* for goals which can be proven.
This would allow for overlapping impls with potentially different associated items,
breaking a bunch of other invariants.

This invariant is currently broken in many different ways while actually something we rely on.
We have to be careful as it is quite easy to break:
- generalization of aliases
- generalization during subtyping binders (luckily not exploitable in coherence)

### Trait solving must not depend on lifetimes being different ✅

If a goal holds with lifetimes being different, it must also hold with these lifetimes being the same. We otherwise get post-monomorphization errors during codegen or unsoundness due to invalid vtables.

We could also just get inconsistent behavior when first proving a goal with different lifetimes which are later constrained to be equal.

### Trait solving in bodies must not depend on lifetimes being equal ✅

We also have to be careful with relying on equality of regions in the trait solver.
This is fine for codegen, as we treat all erased regions as equal. We can however
lose equality information from HIR to MIR typeck.

This currently does not hold with the new solver: [trait-system-refactor-initiative#27](https://github.com/rust-lang/trait-system-refactor-initiative/issues/27).

### Removing ambiguity makes strictly more things compile ❌

Ideally we *should* not rely on ambiguity for things to compile.
Not doing that will cause future improvements to be breaking changes.

Due to *incompleteness* this is not the case,
and improving inference can result in inference changes, breaking existing projects.

### Semantic equality implies structural equality ✅

Two types being equal in the type system must mean that they have the
same `TypeId` after instantiating their generic parameters with concrete
arguments. We can otherwise use their different `TypeId`s to impact trait selection.

We lookup types using structural equality during codegen, but this shouldn't necessarily be unsound
- may result in redundant method codegen or backend type check errors?
- we also rely on it in CTFE assertions

### Semantically different types have different `TypeId`s ✅

Semantically different `'static` types need different `TypeId`s to avoid transmutes,
for example `for<'a> fn(&'a str)` vs `fn(&'static str)` must have a different `TypeId`.


[coherence chapter]: ../coherence.md
