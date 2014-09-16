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
  on the impl itself.

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

One important point is that candidate assembly considers *only the
input types* of the obligation when deciding whether an impl applies
or not. Consider the following example:

    trait Convert<T> { // T is output, Self is input
        fn convert(&self) -> T;
    }

    impl Convert<uint> for int { ... }

Now assume we have an obligation `int : Convert<char>`. During
candidate assembly, the impl above would be considered a definitively
applicable candidate, because it has the same self type (`int`). The
fact that the output type parameter `T` is `uint` on the impl and
`char` in the obligation is not considered.

#### Skolemization

We (at least currently) wish to guarantee "crate concatenability" --
which basically means that you could take two crates, concatenate
them textually, and the combined crate would continue to compile. The
only real way that this relates to trait matching is with
inference. We have to be careful not to influence unbound type
variables during the selection process, basically.

Here is an example:

    trait Foo { fn method() { ... }}
    impl Foo for int { ... }

    fn something() {
        let mut x = None; // `x` has type `Option<?>`
        loop {
            match x {
                Some(ref y) => { // `y` has type ?
                    y.method();  // (*)
                    ...
        }}}
    }

The question is, can we resolve the call to `y.method()`? We don't yet
know what type `y` has. However, there is only one impl in scope, and
it is for `int`, so perhaps we could deduce that `y` *must* have type
`int` (and hence the type of `x` is `Option<int>`)? This is actually
sound reasoning: `int` is the only type in scope that could possibly
make this program type check. However, this deduction is a bit
"unstable", though, because if we concatenated with another crate that
defined a newtype and implemented `Foo` for this newtype, then the
inference would fail, because there would be two potential impls, not
one.

It is unclear how important this property is. It might be nice to drop it.
But for the time being we maintain it.

The way we do this is by *skolemizing* the obligation self type during
the selection process -- skolemizing means, basically, replacing all
unbound type variables with a new "skolemized" type. Each skolemized
type is basically considered "as if" it were some fresh type that is
distinct from all other types. The skolemization process also replaces
lifetimes with `'static`, see the section on lifetimes below for an
explanation.

In the example above, this means that when matching `y.method()` we
would convert the type of `y` from a type variable `?` to a skolemized
type `X`. Then, since `X` cannot unify with `int`, the match would
fail.  Special code exists to check that the match failed because a
skolemized type could not be unified with another kind of type -- this is
not considered a definitive failure, but rather an ambiguous result,
since if the type variable were later to be unified with int, then this
obligation could be resolved then.

*Note:* Currently, method matching does not use the trait resolution
code, so if you in fact type in the example above, it may
compile. Hopefully this will be fixed in later patches.

#### Matching

The subroutines that decide whether a particular impl/where-clause/etc
applies to a particular obligation. At the moment, this amounts to
unifying the self types, but in the future we may also recursively
consider some of the nested obligations, in the case of an impl.

#### Lifetimes and selection

Because of how that lifetime inference works, it is not possible to
give back immediate feedback as to whether a unification or subtype
relationship between lifetimes holds or not. Therefore, lifetime
matching is *not* considered during selection. This is achieved by
having the skolemization process just replace *ALL* lifetimes with
`'static`. Later, during confirmation, the non-skolemized self-type
will be unified with the type from the impl (or whatever). This may
yield lifetime constraints that will later be found to be in error (in
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
