// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# TRAIT RESOLUTION

This document describes the general process and points out some non-obvious
things.

## Major concepts

Trait resolution is the process of pairing up an impl with each
reference to a trait. So, for example, if there is a generic function like:

    fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> { ... }

and then a call to that function:

    let v: Vec<int> = clone_slice([1, 2, 3].as_slice())

it is the job of trait resolution to figure out (in which case)
whether there exists an impl of `int : Clone`

Note that in some cases, like generic functions, we may not be able to
find a specific impl, but we can figure out that the caller must
provide an impl. To see what I mean, consider the body of `clone_slice`:

    fn clone_slice<T:Clone>(x: &[T]) -> Vec<T> {
        let mut v = Vec::new();
        for e in x.iter() {
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
wanted to permit conversion between `int` and `uint`, we might
implement `Convert` like so:

```rust
impl Convert<uint> for int { ... } // int -> uint
impl Convert<int> for uint { ... } // uint -> uint
```

Now imagine there is some code like the following:

```rust
let x: int = ...;
let y = x.convert();
```

The call to convert will generate a trait reference `Convert<$Y> for
int`, where `$Y` is the type variable representing the type of
`y`. When we match this against the two impls we can see, we will find
that only one remains: `Convert<uint> for int`. Therefore, we can
select this impl, which will cause the type of `$Y` to be unified to
`uint`. (Note that while assembling candidates, we do the initial
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
impl specified that `T` would be `uint`, but the obligation reported
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
we do not have the full substitutions in terms of the type varibales of the impl available
to us, so we must run trait selection to figure everything out.

Here is an example:

    trait Foo { ... }
    impl<U,T:Bar<U>> Foo for Vec<T> { ... }

    impl Bar<uint> for int { ... }

After one shallow round of selection for an obligation like `Vec<int>
: Foo`, we would know which impl we want, and we would know that
`T=int`, but we do not know the type of `U`.  We must select the
nested obligation `int : Bar<U>` to find out that `U=uint`.

It would be good to only do *just as much* nested resolution as
necessary. Currently, though, we just do a full resolution.

*/
