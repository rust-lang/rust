// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Type inference engine

This is loosely based on standard HM-type inference, but with an
extension to try and accommodate subtyping.  There is nothing
principled about this extension; it's sound---I hope!---but it's a
heuristic, ultimately, and does not guarantee that it finds a valid
typing even if one exists (in fact, there are known scenarios where it
fails, some of which may eventually become problematic).

## Key idea

The main change is that each type variable T is associated with a
lower-bound L and an upper-bound U.  L and U begin as bottom and top,
respectively, but gradually narrow in response to new constraints
being introduced.  When a variable is finally resolved to a concrete
type, it can (theoretically) select any type that is a supertype of L
and a subtype of U.

There are several critical invariants which we maintain:

- the upper-bound of a variable only becomes lower and the lower-bound
  only becomes higher over time;
- the lower-bound L is always a subtype of the upper bound U;
- the lower-bound L and upper-bound U never refer to other type variables,
  but only to types (though those types may contain type variables).

> An aside: if the terms upper- and lower-bound confuse you, think of
> "supertype" and "subtype".  The upper-bound is a "supertype"
> (super=upper in Latin, or something like that anyway) and the lower-bound
> is a "subtype" (sub=lower in Latin).  I find it helps to visualize
> a simple class hierarchy, like Java minus interfaces and
> primitive types.  The class Object is at the root (top) and other
> types lie in between.  The bottom type is then the Null type.
> So the tree looks like:
>
>             Object
>             /    \
>         String   Other
>             \    /
>             (null)
>
> So the upper bound type is the "supertype" and the lower bound is the
> "subtype" (also, super and sub mean upper and lower in Latin, or something
> like that anyway).

## Satisfying constraints

At a primitive level, there is only one form of constraint that the
inference understands: a subtype relation.  So the outside world can
say "make type A a subtype of type B".  If there are variables
involved, the inferencer will adjust their upper- and lower-bounds as
needed to ensure that this relation is satisfied. (We also allow "make
type A equal to type B", but this is translated into "A <: B" and "B
<: A")

As stated above, we always maintain the invariant that type bounds
never refer to other variables.  This keeps the inference relatively
simple, avoiding the scenario of having a kind of graph where we have
to pump constraints along and reach a fixed point, but it does impose
some heuristics in the case where the user is relating two type
variables A <: B.

Combining two variables such that variable A will forever be a subtype
of variable B is the trickiest part of the algorithm because there is
often no right choice---that is, the right choice will depend on
future constraints which we do not yet know. The problem comes about
because both A and B have bounds that can be adjusted in the future.
Let's look at some of the cases that can come up.

Imagine, to start, the best case, where both A and B have an upper and
lower bound (that is, the bounds are not top nor bot respectively). In
that case, if we're lucky, A.ub <: B.lb, and so we know that whatever
A and B should become, they will forever have the desired subtyping
relation.  We can just leave things as they are.

### Option 1: Unify

However, suppose that A.ub is *not* a subtype of B.lb.  In
that case, we must make a decision.  One option is to unify A
and B so that they are one variable whose bounds are:

    UB = GLB(A.ub, B.ub)
    LB = LUB(A.lb, B.lb)

(Note that we will have to verify that LB <: UB; if it does not, the
types are not intersecting and there is an error) In that case, A <: B
holds trivially because A==B.  However, we have now lost some
flexibility, because perhaps the user intended for A and B to end up
as different types and not the same type.

Pictorally, what this does is to take two distinct variables with
(hopefully not completely) distinct type ranges and produce one with
the intersection.

                      B.ub                  B.ub
                       /\                    /
               A.ub   /  \           A.ub   /
               /   \ /    \              \ /
              /     X      \              UB
             /     / \      \            / \
            /     /   /      \          /   /
            \     \  /       /          \  /
             \      X       /             LB
              \    / \     /             / \
               \  /   \   /             /   \
               A.lb    B.lb          A.lb    B.lb


### Option 2: Relate UB/LB

Another option is to keep A and B as distinct variables but set their
bounds in such a way that, whatever happens, we know that A <: B will hold.
This can be achieved by ensuring that A.ub <: B.lb.  In practice there
are two ways to do that, depicted pictorally here:

        Before                Option #1            Option #2

                 B.ub                B.ub                B.ub
                  /\                 /  \                /  \
          A.ub   /  \        A.ub   /(B')\       A.ub   /(B')\
          /   \ /    \           \ /     /           \ /     /
         /     X      \         __UB____/             UB    /
        /     / \      \       /  |                   |    /
       /     /   /      \     /   |                   |   /
       \     \  /       /    /(A')|                   |  /
        \      X       /    /     LB            ______LB/
         \    / \     /    /     / \           / (A')/ \
          \  /   \   /     \    /   \          \    /   \
          A.lb    B.lb       A.lb    B.lb        A.lb    B.lb

In these diagrams, UB and LB are defined as before.  As you can see,
the new ranges `A'` and `B'` are quite different from the range that
would be produced by unifying the variables.

### What we do now

Our current technique is to *try* (transactionally) to relate the
existing bounds of A and B, if there are any (i.e., if `UB(A) != top
&& LB(B) != bot`).  If that succeeds, we're done.  If it fails, then
we merge A and B into same variable.

This is not clearly the correct course.  For example, if `UB(A) !=
top` but `LB(B) == bot`, we could conceivably set `LB(B)` to `UB(A)`
and leave the variables unmerged.  This is sometimes the better
course, it depends on the program.

The main case which fails today that I would like to support is:

    fn foo<T>(x: T, y: T) { ... }

    fn bar() {
        let x: @mut int = @mut 3;
        let y: @int = @3;
        foo(x, y);
    }

In principle, the inferencer ought to find that the parameter `T` to
`foo(x, y)` is `@const int`.  Today, however, it does not; this is
because the type variable `T` is merged with the type variable for
`X`, and thus inherits its UB/LB of `@mut int`.  This leaves no
flexibility for `T` to later adjust to accommodate `@int`.

### What to do when not all bounds are present

In the prior discussion we assumed that A.ub was not top and B.lb was
not bot.  Unfortunately this is rarely the case.  Often type variables
have "lopsided" bounds.  For example, if a variable in the program has
been initialized but has not been used, then its corresponding type
variable will have a lower bound but no upper bound.  When that
variable is then used, we would like to know its upper bound---but we
don't have one!  In this case we'll do different things depending on
how the variable is being used.

## Transactional support

Whenever we adjust merge variables or adjust their bounds, we always
keep a record of the old value.  This allows the changes to be undone.

## Regions

I've only talked about type variables here, but region variables
follow the same principle.  They have upper- and lower-bounds.  A
region A is a subregion of a region B if A being valid implies that B
is valid.  This basically corresponds to the block nesting structure:
the regions for outer block scopes are superregions of those for inner
block scopes.

## Integral and floating-point type variables

There is a third variety of type variable that we use only for
inferring the types of unsuffixed integer literals.  Integral type
variables differ from general-purpose type variables in that there's
no subtyping relationship among the various integral types, so instead
of associating each variable with an upper and lower bound, we just
use simple unification.  Each integer variable is associated with at
most one integer type.  Floating point types are handled similarly to
integral types.

## GLB/LUB

Computing the greatest-lower-bound and least-upper-bound of two
types/regions is generally straightforward except when type variables
are involved. In that case, we follow a similar "try to use the bounds
when possible but otherwise merge the variables" strategy.  In other
words, `GLB(A, B)` where `A` and `B` are variables will often result
in `A` and `B` being merged and the result being `A`.

## Type coercion

We have a notion of assignability which differs somewhat from
subtyping; in particular it may cause region borrowing to occur.  See
the big comment later in this file on Type Coercion for specifics.

### In conclusion

I showed you three ways to relate `A` and `B`.  There are also more,
of course, though I'm not sure if there are any more sensible options.
The main point is that there are various options, each of which
produce a distinct range of types for `A` and `B`.  Depending on what
the correct values for A and B are, one of these options will be the
right choice: but of course we don't know the right values for A and B
yet, that's what we're trying to find!  In our code, we opt to unify
(Option #1).

# Implementation details

We make use of a trait-like impementation strategy to consolidate
duplicated code between subtypes, GLB, and LUB computations.  See the
section on "Type Combining" below for details.

*/