# TRAIT RESOLUTION

This document describes the general process and points out some non-obvious
things.

## Major concepts

Trait resolution is the process of pairing up an impl with each
reference to a trait. So, for example, if there is a generic function like:

    fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> { ... }

and then a call to that function:

    let v: Vec<isize> = clone_slice([1, 2, 3])

it is the job of trait resolution to figure out (in which case)
whether there exists an impl of `isize : Clone`

Note that in some cases, like generic functions, we may not be able to
find a specific impl, but we can figure out that the caller must
provide an impl. To see what I mean, consider the body of `clone_slice`:

    fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> {
        let mut v = Vec::new();
        for e in &x {
            v.push((*e).clone()); // (*)
        }
    }

The line marked `(*)` is only legal if `T` (the type of `*e`)
implements the `Clone` trait. Naturally, since we don't know what `T`
is, we can't find the specific impl; but based on the bound `T:Clone`,
we can say that there exists an impl which the caller must provide.

We use the term *obligation* to refer to a trait reference in need of
an impl.

## Overview

Trait resolution consists of three major parts:

- SELECTION: Deciding how to resolve a specific obligation. For
  example, selection might decide that a specific obligation can be
  resolved by employing an impl which matches the self type, or by
  using a parameter bound. In the case of an impl, Selecting one
  obligation can create *nested obligations* because of where clauses
  on the impl itself. It may also require evaluating those nested
  obligations to resolve ambiguities.

- FULFILLMENT: The fulfillment code is what tracks that obligations
  are completely fulfilled. Basically it is a worklist of obligations
  to be selected: once selection is successful, the obligation is
  removed from the worklist and any nested obligations are enqueued.

- COHERENCE: The coherence checks are intended to ensure that there
  are never overlapping impls, where two impls could be used with
  equal precedence.

## Selection

Selection is the process of deciding whether an obligation can be
resolved and, if so, how it is to be resolved (via impl, where clause, etc).
The main interface is the `select()` function, which takes an obligation
and returns a `SelectionResult`. There are three possible outcomes:

- `Ok(Some(selection))` -- yes, the obligation can be resolved, and
  `selection` indicates how. If the impl was resolved via an impl,
  then `selection` may also indicate nested obligations that are required
  by the impl.

- `Ok(None)` -- we are not yet sure whether the obligation can be
  resolved or not. This happens most commonly when the obligation
  contains unbound type variables.

- `Err(err)` -- the obligation definitely cannot be resolved due to a
  type error, or because there are no impls that could possibly apply,
  etc.

The basic algorithm for selection is broken into two big phases:
candidate assembly and confirmation.

### Candidate assembly

Searches for impls/where-clauses/etc that might
possibly be used to satisfy the obligation. Each of those is called
a candidate. To avoid ambiguity, we want to find exactly one
candidate that is definitively applicable. In some cases, we may not
know whether an impl/where-clause applies or not -- this occurs when
the obligation contains unbound inference variables.

The basic idea for candidate assembly is to do a first pass in which
we identify all possible candidates. During this pass, all that we do
is try and unify the type parameters. (In particular, we ignore any
nested where clauses.) Presuming that this unification succeeds, the
impl is added as a candidate.

Once this first pass is done, we can examine the set of candidates. If
it is a singleton set, then we are done: this is the only impl in
scope that could possibly apply. Otherwise, we can winnow down the set
of candidates by using where clauses and other conditions. If this
reduced set yields a single, unambiguous entry, we're good to go,
otherwise the result is considered ambiguous.

#### The basic process: Inferring based on the impls we see

This process is easier if we work through some examples. Consider
the following trait:

```
trait Convert<Target> {
    fn convert(&self) -> Target;
}
```

This trait just has one method. It's about as simple as it gets. It
converts from the (implicit) `Self` type to the `Target` type. If we
wanted to permit conversion between `isize` and `usize`, we might
implement `Convert` like so:

```rust
impl Convert<usize> for isize { ... } // isize -> usize
impl Convert<isize> for usize { ... } // usize -> isize
```

Now imagine there is some code like the following:

```rust
let x: isize = ...;
let y = x.convert();
```

The call to convert will generate a trait reference `Convert<$Y> for
isize`, where `$Y` is the type variable representing the type of
`y`. When we match this against the two impls we can see, we will find
that only one remains: `Convert<usize> for isize`. Therefore, we can
select this impl, which will cause the type of `$Y` to be unified to
`usize`. (Note that while assembling candidates, we do the initial
unifications in a transaction, so that they don't affect one another.)

There are tests to this effect in src/test/run-pass:

   traits-multidispatch-infer-convert-source-and-target.rs
   traits-multidispatch-infer-convert-target.rs

#### Winnowing: Resolving ambiguities

But what happens if there are multiple impls where all the types
unify? Consider this example:

```rust
trait Get {
    fn get(&self) -> Self;
}

impl<T:Copy> Get for T {
    fn get(&self) -> T { *self }
}

impl<T:Get> Get for Box<T> {
    fn get(&self) -> Box<T> { box get_it(&**self) }
}
```

What happens when we invoke `get_it(&box 1_u16)`, for example? In this
case, the `Self` type is `Box<u16>` -- that unifies with both impls,
because the first applies to all types, and the second to all
boxes. In the olden days we'd have called this ambiguous. But what we
do now is do a second *winnowing* pass that considers where clauses
and attempts to remove candidates -- in this case, the first impl only
applies if `Box<u16> : Copy`, which doesn't hold. After winnowing,
then, we are left with just one candidate, so we can proceed. There is
a test of this in `src/test/run-pass/traits-conditional-dispatch.rs`.

#### Matching

The subroutines that decide whether a particular impl/where-clause/etc
applies to a particular obligation. At the moment, this amounts to
unifying the self types, but in the future we may also recursively
consider some of the nested obligations, in the case of an impl.

#### Lifetimes and selection

Because of how that lifetime inference works, it is not possible to
give back immediate feedback as to whether a unification or subtype
relationship between lifetimes holds or not. Therefore, lifetime
matching is *not* considered during selection. This is reflected in
the fact that subregion assignment is infallible. This may yield
lifetime constraints that will later be found to be in error (in
contrast, the non-lifetime-constraints have already been checked
during selection and can never cause an error, though naturally they
may lead to other errors downstream).

#### Where clauses

Besides an impl, the other major way to resolve an obligation is via a
where clause. The selection process is always given a *parameter
environment* which contains a list of where clauses, which are
basically obligations that can assume are satisfiable. We will iterate
over that list and check whether our current obligation can be found
in that list, and if so it is considered satisfied. More precisely, we
want to check whether there is a where-clause obligation that is for
the same trait (or some subtrait) and for which the self types match,
using the definition of *matching* given above.

Consider this simple example:

     trait A1 { ... }
     trait A2 : A1 { ... }

     trait B { ... }

     fn foo<X:A2+B> { ... }

Clearly we can use methods offered by `A1`, `A2`, or `B` within the
body of `foo`. In each case, that will incur an obligation like `X :
A1` or `X : A2`. The parameter environment will contain two
where-clauses, `X : A2` and `X : B`. For each obligation, then, we
search this list of where-clauses.  To resolve an obligation `X:A1`,
we would note that `X:A2` implies that `X:A1`.

### Confirmation

Confirmation unifies the output type parameters of the trait with the
values found in the obligation, possibly yielding a type error.  If we
return to our example of the `Convert` trait from the previous
section, confirmation is where an error would be reported, because the
impl specified that `T` would be `usize`, but the obligation reported
`char`. Hence the result of selection would be an error.

### Selection during translation

During type checking, we do not store the results of trait selection.
We simply wish to verify that trait selection will succeed. Then
later, at trans time, when we have all concrete types available, we
can repeat the trait selection.  In this case, we do not consider any
where-clauses to be in scope. We know that therefore each resolution
will resolve to a particular impl.

One interesting twist has to do with nested obligations. In general, in trans,
we only need to do a "shallow" selection for an obligation. That is, we wish to
identify which impl applies, but we do not (yet) need to decide how to select
any nested obligations. Nonetheless, we *do* currently do a complete resolution,
and that is because it can sometimes inform the results of type inference. That is,
we do not have the full substitutions in terms of the type variables of the impl available
to us, so we must run trait selection to figure everything out.

Here is an example:

    trait Foo { ... }
    impl<U,T:Bar<U>> Foo for Vec<T> { ... }

    impl Bar<usize> for isize { ... }

After one shallow round of selection for an obligation like `Vec<isize>
: Foo`, we would know which impl we want, and we would know that
`T=isize`, but we do not know the type of `U`.  We must select the
nested obligation `isize : Bar<U>` to find out that `U=usize`.

It would be good to only do *just as much* nested resolution as
necessary. Currently, though, we just do a full resolution.

# Higher-ranked trait bounds

One of the more subtle concepts at work are *higher-ranked trait
bounds*. An example of such a bound is `for<'a> MyTrait<&'a isize>`.
Let's walk through how selection on higher-ranked trait references
works.

## Basic matching and skolemization leaks

Let's walk through the test `compile-fail/hrtb-just-for-static.rs` to see
how it works. The test starts with the trait `Foo`:

```rust
trait Foo<X> {
    fn foo(&self, x: X) { }
}
```

Let's say we have a function `want_hrtb` that wants a type which
implements `Foo<&'a isize>` for any `'a`:

```rust
fn want_hrtb<T>() where T : for<'a> Foo<&'a isize> { ... }
```

Now we have a struct `AnyInt` that implements `Foo<&'a isize>` for any
`'a`:

```rust
struct AnyInt;
impl<'a> Foo<&'a isize> for AnyInt { }
```

And the question is, does `AnyInt : for<'a> Foo<&'a isize>`? We want the
answer to be yes. The algorithm for figuring it out is closely related
to the subtyping for higher-ranked types (which is described in
`middle::infer::higher_ranked::doc`, but also in a [paper by SPJ] that
I recommend you read).

1. Skolemize the obligation.
2. Match the impl against the skolemized obligation.
3. Check for skolemization leaks.

[paper by SPJ]: http://research.microsoft.com/en-us/um/people/simonpj/papers/higher-rank/

So let's work through our example. The first thing we would do is to
skolemize the obligation, yielding `AnyInt : Foo<&'0 isize>` (here `'0`
represents skolemized region #0). Note that now have no quantifiers;
in terms of the compiler type, this changes from a `ty::PolyTraitRef`
to a `TraitRef`. We would then create the `TraitRef` from the impl,
using fresh variables for it's bound regions (and thus getting
`Foo<&'$a isize>`, where `'$a` is the inference variable for `'a`). Next
we relate the two trait refs, yielding a graph with the constraint
that `'0 == '$a`. Finally, we check for skolemization "leaks" -- a
leak is basically any attempt to relate a skolemized region to another
skolemized region, or to any region that pre-existed the impl match.
The leak check is done by searching from the skolemized region to find
the set of regions that it is related to in any way. This is called
the "taint" set. To pass the check, that set must consist *solely* of
itself and region variables from the impl. If the taint set includes
any other region, then the match is a failure. In this case, the taint
set for `'0` is `{'0, '$a}`, and hence the check will succeed.

Let's consider a failure case. Imagine we also have a struct

```rust
struct StaticInt;
impl Foo<&'static isize> for StaticInt;
```

We want the obligation `StaticInt : for<'a> Foo<&'a isize>` to be
considered unsatisfied. The check begins just as before. `'a` is
skolemized to `'0` and the impl trait reference is instantiated to
`Foo<&'static isize>`. When we relate those two, we get a constraint
like `'static == '0`. This means that the taint set for `'0` is `{'0,
'static}`, which fails the leak check.

## Higher-ranked trait obligations

Once the basic matching is done, we get to another interesting topic:
how to deal with impl obligations. I'll work through a simple example
here. Imagine we have the traits `Foo` and `Bar` and an associated impl:

```
trait Foo<X> {
    fn foo(&self, x: X) { }
}

trait Bar<X> {
    fn bar(&self, x: X) { }
}

impl<X,F> Foo<X> for F
    where F : Bar<X>
{
}
```

Now let's say we have a obligation `for<'a> Foo<&'a isize>` and we match
this impl. What obligation is generated as a result? We want to get
`for<'a> Bar<&'a isize>`, but how does that happen?

After the matching, we are in a position where we have a skolemized
substitution like `X => &'0 isize`. If we apply this substitution to the
impl obligations, we get `F : Bar<&'0 isize>`. Obviously this is not
directly usable because the skolemized region `'0` cannot leak out of
our computation.

What we do is to create an inverse mapping from the taint set of `'0`
back to the original bound region (`'a`, here) that `'0` resulted
from. (This is done in `higher_ranked::plug_leaks`). We know that the
leak check passed, so this taint set consists solely of the skolemized
region itself plus various intermediate region variables. We then walk
the trait-reference and convert every region in that taint set back to
a late-bound region, so in this case we'd wind up with `for<'a> F :
Bar<&'a isize>`.

# Caching and subtle considerations therewith

In general we attempt to cache the results of trait selection.  This
is a somewhat complex process. Part of the reason for this is that we
want to be able to cache results even when all the types in the trait
reference are not fully known. In that case, it may happen that the
trait selection process is also influencing type variables, so we have
to be able to not only cache the *result* of the selection process,
but *replay* its effects on the type variables.

## An example

The high-level idea of how the cache works is that we first replace
all unbound inference variables with skolemized versions. Therefore,
if we had a trait reference `usize : Foo<$1>`, where `$n` is an unbound
inference variable, we might replace it with `usize : Foo<%0>`, where
`%n` is a skolemized type. We would then look this up in the cache.
If we found a hit, the hit would tell us the immediate next step to
take in the selection process: i.e., apply impl #22, or apply where
clause `X : Foo<Y>`. Let's say in this case there is no hit.
Therefore, we search through impls and where clauses and so forth, and
we come to the conclusion that the only possible impl is this one,
with def-id 22:

    impl Foo<isize> for usize { ... } // Impl #22

We would then record in the cache `usize : Foo<%0> ==>
ImplCandidate(22)`. Next we would confirm `ImplCandidate(22)`, which
would (as a side-effect) unify `$1` with `isize`.

Now, at some later time, we might come along and see a `usize :
Foo<$3>`.  When skolemized, this would yield `usize : Foo<%0>`, just as
before, and hence the cache lookup would succeed, yielding
`ImplCandidate(22)`. We would confirm `ImplCandidate(22)` which would
(as a side-effect) unify `$3` with `isize`.

## Where clauses and the local vs global cache

One subtle interaction is that the results of trait lookup will vary
depending on what where clauses are in scope. Therefore, we actually
have *two* caches, a local and a global cache. The local cache is
attached to ParamEnv` and the global cache attached
to the `tcx`. We use the local cache whenever the result might depend
on the where clauses that are in scope. The determination of which
cache to use is done by the method `pick_candidate_cache` in
`select.rs`. At the moment, we use a very simple, conservative rule:
if there are any where-clauses in scope, then we use the local cache.
We used to try and draw finer-grained distinctions, but that led to a
serious of annoying and weird bugs like #22019 and #18290. This simple
rule seems to be pretty clearly safe and also still retains a very
high hit rate (~95% when compiling rustc).

# Specialization

Defined in the `specialize` module.

The basic strategy is to build up a *specialization graph* during
coherence checking. Insertion into the graph locates the right place
to put an impl in the specialization hierarchy; if there is no right
place (due to partial overlap but no containment), you get an overlap
error. Specialization is consulted when selecting an impl (of course),
and the graph is consulted when propagating defaults down the
specialization hierarchy.

You might expect that the specialization graph would be used during
selection -- i.e., when actually performing specialization. This is
not done for two reasons:

- It's merely an optimization: given a set of candidates that apply,
  we can determine the most specialized one by comparing them directly
  for specialization, rather than consulting the graph. Given that we
  also cache the results of selection, the benefit of this
  optimization is questionable.

- To build the specialization graph in the first place, we need to use
  selection (because we need to determine whether one impl specializes
  another). Dealing with this reentrancy would require some additional
  mode switch for selection. Given that there seems to be no strong
  reason to use the graph anyway, we stick with a simpler approach in
  selection, and use the graph only for propagating default
  implementations.

Trait impl selection can succeed even when multiple impls can apply,
as long as they are part of the same specialization family. In that
case, it returns a *single* impl on success -- this is the most
specialized impl *known* to apply. However, if there are any inference
variables in play, the returned impl may not be the actual impl we
will use at trans time. Thus, we take special care to avoid projecting
associated types unless either (1) the associated type does not use
`default` and thus cannot be overridden or (2) all input types are
known concretely.
