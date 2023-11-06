# Invariants of the type system

FIXME: This file talks about invariants of the type system as a whole, not only the solver

There are a lot of invariants - things the type system guarantees to be true at all times -
which are desirable or expected from other languages and type systems. Unfortunately, quite
a few of them do not hold in Rust right now. This is either a fundamental to its design or
caused by bugs and something that may change in the future.

It is important to know about the things you can assume while working on - and with - the
type system, so here's an incomplete and inofficial list of invariants of
the core type system:

- ✅: this invariant mostly holds, with some weird exceptions, you can rely on it outside
of these cases
- ❌: this invariant does not hold, either due to bugs or by design, you must not rely on
it for soundness or have to be incredibly careful when doing so

### `wf(X)` implies `wf(normalize(X))` ✅

If a type containing aliases is well-formed, it should also be
well-formed after normalizing said aliases. We rely on this as
otherwise we would have to re-check for well-formedness for these
types.

This is unfortunately broken for `<fndef as FnOnce<..>>::Output` due to implied bounds,
resulting in [#114936].

### Structural equality modulo regions implies semantic equality ✅

If you have a some type and equate it to itself after replacing any regions with unique
inference variables in both the lhs and rhs, the now potentially structurally different
types should still be equal to each other.

Needed to prevent goals from succeeding in HIR typeck and then failing in MIR borrowck.
If this does invariant is broken MIR typeck ends up failing with an ICE.

### Applying inference results from a goal does not change its result ❌

TODO: this invariant is formulated in a weird way and needs to be elaborated.
Pretty much: I would like this check to only fail if there's a solver bug:
https://github.com/rust-lang/rust/blob/2ffeb4636b4ae376f716dc4378a7efb37632dc2d/compiler/rustc_trait_selection/src/solve/eval_ctxt.rs#L391-L407

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

### Generic goals and their instantiations have the same result ✅

Pretty much: If we successfully typecheck a generic function concrete instantiations
of that function should also typeck. We should not get errors post-monomorphization.
We can however get overflow errors at that point.

TODO: example for overflow error post-monomorphization

This invariant is relied on to allow the normalization of generic aliases. Breaking
it can easily result in unsoundness, e.g. [#57893](https://github.com/rust-lang/rust/issues/57893)

### Trait goals in empty environments are proven by a unique impl ✅

If a trait goal holds with an empty environment, there should be a unique `impl`,
either user-defined or builtin, which is used to prove that goal. This is
necessary to select a unique method. It 

We do however break this invariant in few cases, some of which are due to bugs,
some by design:
- *marker traits* are allowed to overlap as they do not have associated items
- *specialization* allows specializing impls to overlap with their parent
- the builtin trait object trait implementation can overlap with a user-defined impl:
[#57893]

### The type system is complete ❌

The type system is not complete, it often adds unnecessary inference constraints, and errors
even though the goal could hold.

- method selection
- opaque type inference
- handling type outlives constraints
- preferring `ParamEnv` candidates over `Impl` candidates during candidate selection
in the trait solver

#### The type system is complete during the implicit negative overlap check in coherence ✅

During the implicit negative overlap check in coherence we must never return *error* for
goals which can be proven. This would allow for overlapping impls with potentially different
associated items, breaking a bunch of other invariants.

This invariant is currently broken in many different ways while actually something we rely on.
We have to be careful as it is quite easy to break:
- generalization of aliases
- generalization during subtyping binders (luckily not exploitable in coherence)

### Trait solving must be (free) lifetime agnostic ✅

Trait solving during codegen should have the same result as during typeck. As we erase
all free regions during codegen we must not rely on them during typeck. A noteworthy example
is special behavior for `'static`.

We also have to be careful with relying on equality of regions in the trait solver.
This is fine for codegen, as we treat all erased regions as equal. We can however
lose equality information from HIR to MIR typeck.

The new solver "uniquifies regions" during canonicalization, canonicalizing `u32: Trait<'x, 'x>`
as `exists<'0, '1> u32: Trait<'0, '1>`, to make it harder to rely on this property.

### Removing ambiguity makes strictly more things compile ❌

Ideally we *should* not rely on ambiguity for things to compile.
Not doing that will cause future improvements to be breaking changes.

Due to *incompleteness* this is not the case and improving inference can result in inference
changes, breaking existing projects.

### Semantic equality implies structural equality ✅

Two types being equal in the type system must mean that they have the
same `TypeId` after instantiating their generic parameters with concrete
arguments. This currently does not hold: [#97156].

[#57893]: https://github.com/rust-lang/rust/issues/57893
[#97156]: https://github.com/rust-lang/rust/issues/97156
[#114936]: https://github.com/rust-lang/rust/issues/114936