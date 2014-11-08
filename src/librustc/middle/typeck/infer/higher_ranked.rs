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
 * Helper routines for higher-ranked things. See the `doc` module at
 * the end of the file for details.
 */

use middle::ty;
use middle::ty::replace_late_bound_regions;
use middle::typeck::infer::combine;
use middle::typeck::infer::combine::Combine;
use middle::typeck::infer::{cres};
use middle::typeck::infer::fold_regions_in;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::region_inference::{RegionMark};
use middle::ty_fold::TypeFoldable;
use std::collections::HashMap;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::{bound_region_to_string, Repr};

pub trait HigherRanked : TypeFoldable + Repr {
    fn binder_id(&self) -> ast::NodeId;
    fn super_combine<'tcx,C:Combine<'tcx>>(combine: &C, a: &Self, b: &Self) -> cres<Self>;
}

pub trait HigherRankedRelations {
    fn higher_ranked_sub<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked;

    fn higher_ranked_lub<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked;

    fn higher_ranked_glb<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked;
}

impl<'tcx,C> HigherRankedRelations for C
    where C : Combine<'tcx>
{
    fn higher_ranked_sub<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked
    {
        debug!("higher_ranked_sub(a={}, b={})",
               a.repr(self.tcx()), b.repr(self.tcx()));

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment at the end of the file in the (inlined) module
        // `doc`.

        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.infcx().region_vars.mark();

        // First, we instantiate each bound region in the subtype with a fresh
        // region variable.
        let (a_prime, _) =
            self.infcx().replace_late_bound_regions_with_fresh_regions(
                self.trace().origin.span(), a.binder_id(), a);

        // Second, we instantiate each bound region in the supertype with a
        // fresh concrete region.
        let (b_prime, skol_map) = {
            replace_late_bound_regions(self.tcx(), b.binder_id(), b, |br| {
                let skol = self.infcx().region_vars.new_skolemized(br);
                debug!("Bound region {} skolemized to {}",
                       bound_region_to_string(self.tcx(), "", false, br),
                       skol);
                skol
            })
        };

        debug!("a_prime={}", a_prime.repr(self.tcx()));
        debug!("b_prime={}", b_prime.repr(self.tcx()));

        // Compare types now that bound regions have been replaced.
        let sig = try!(HigherRanked::super_combine(self, &a_prime, &b_prime));

        // Presuming type comparison succeeds, we need to check
        // that the skolemized regions do not "leak".
        let new_vars =
            self.infcx().region_vars.vars_created_since_mark(mark);
        for (&skol_br, &skol) in skol_map.iter() {
            let tainted = self.infcx().region_vars.tainted(mark, skol);
            for tainted_region in tainted.iter() {
                // Each skolemized should only be relatable to itself
                // or new variables:
                match *tainted_region {
                    ty::ReInfer(ty::ReVar(ref vid)) => {
                        if new_vars.iter().any(|x| x == vid) { continue; }
                    }
                    _ => {
                        if *tainted_region == skol { continue; }
                    }
                };

                // A is not as polymorphic as B:
                if self.a_is_expected() {
                    debug!("Not as polymorphic!");
                    return Err(ty::terr_regions_insufficiently_polymorphic(
                        skol_br, *tainted_region));
                } else {
                    debug!("Overly polymorphic!");
                    return Err(ty::terr_regions_overly_polymorphic(
                        skol_br, *tainted_region));
                }
            }
        }

        return Ok(sig);
    }

    fn higher_ranked_lub<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked
    {
        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.infcx().region_vars.mark();

        // Instantiate each bound region with a fresh region variable.
        let (a_with_fresh, a_map) =
            self.infcx().replace_late_bound_regions_with_fresh_regions(
                self.trace().origin.span(), a.binder_id(), a);
        let (b_with_fresh, _) =
            self.infcx().replace_late_bound_regions_with_fresh_regions(
                self.trace().origin.span(), b.binder_id(), b);

        // Collect constraints.
        let result0 = try!(HigherRanked::super_combine(self, &a_with_fresh, &b_with_fresh));
        debug!("sig0 = {}", result0.repr(self.tcx()));

        // Generalize the regions appearing in sig0 if possible
        let new_vars = self.infcx().region_vars.vars_created_since_mark(mark);
        let span = self.trace().origin.span();
        let result1 =
            fold_regions_in(
                self.tcx(),
                &result0,
                |r| generalize_region(self.infcx(), span, mark, new_vars.as_slice(),
                                      result0.binder_id(), &a_map, r));
        return Ok(result1);

        fn generalize_region(infcx: &InferCtxt,
                             span: Span,
                             mark: RegionMark,
                             new_vars: &[ty::RegionVid],
                             new_scope: ast::NodeId,
                             a_map: &HashMap<ty::BoundRegion, ty::Region>,
                             r0: ty::Region)
                             -> ty::Region {
            // Regions that pre-dated the LUB computation stay as they are.
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                debug!("generalize_region(r0={}): not new variable", r0);
                return r0;
            }

            let tainted = infcx.region_vars.tainted(mark, r0);

            // Variables created during LUB computation which are
            // *related* to regions that pre-date the LUB computation
            // stay as they are.
            if !tainted.iter().all(|r| is_var_in_set(new_vars, *r)) {
                debug!("generalize_region(r0={}): \
                        non-new-variables found in {}",
                       r0, tainted);
                assert!(!r0.is_bound());
                return r0;
            }

            // Otherwise, the variable must be associated with at
            // least one of the variables representing bound regions
            // in both A and B.  Replace the variable with the "first"
            // bound region from A that we find it to be associated
            // with.
            for (a_br, a_r) in a_map.iter() {
                if tainted.iter().any(|x| x == a_r) {
                    debug!("generalize_region(r0={}): \
                            replacing with {}, tainted={}",
                           r0, *a_br, tainted);
                    return ty::ReLateBound(new_scope, *a_br);
                }
            }

            infcx.tcx.sess.span_bug(
                span,
                format!("region {} is not associated with \
                         any bound region from A!",
                        r0).as_slice())
        }
    }

    fn higher_ranked_glb<T>(&self, a: &T, b: &T) -> cres<T>
        where T : HigherRanked
    {
        debug!("{}.higher_ranked_glb({}, {})",
               self.tag(), a.repr(self.tcx()), b.repr(self.tcx()));

        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.infcx().region_vars.mark();

        // Instantiate each bound region with a fresh region variable.
        let (a_with_fresh, a_map) =
            self.infcx().replace_late_bound_regions_with_fresh_regions(
                self.trace().origin.span(), a.binder_id(), a);
        let a_vars = var_ids(self, &a_map);
        let (b_with_fresh, b_map) =
            self.infcx().replace_late_bound_regions_with_fresh_regions(
                self.trace().origin.span(), b.binder_id(), b);
        let b_vars = var_ids(self, &b_map);

        // Collect constraints.
        let result0 = try!(HigherRanked::super_combine(self, &a_with_fresh, &b_with_fresh));
        debug!("result0 = {}", result0.repr(self.tcx()));

        // Generalize the regions appearing in fn_ty0 if possible
        let new_vars = self.infcx().region_vars.vars_created_since_mark(mark);
        let span = self.trace().origin.span();
        let result1 =
            fold_regions_in(
                self.tcx(),
                &result0,
                |r| generalize_region(self.infcx(),
                                      span,
                                      mark,
                                      new_vars.as_slice(),
                                      result0.binder_id(),
                                      &a_map,
                                      a_vars.as_slice(),
                                      b_vars.as_slice(),
                                      r));
        debug!("result1 = {}", result1.repr(self.tcx()));
        return Ok(result1);

        fn generalize_region(infcx: &InferCtxt,
                             span: Span,
                             mark: RegionMark,
                             new_vars: &[ty::RegionVid],
                             new_binder_id: ast::NodeId,
                             a_map: &HashMap<ty::BoundRegion, ty::Region>,
                             a_vars: &[ty::RegionVid],
                             b_vars: &[ty::RegionVid],
                             r0: ty::Region) -> ty::Region {
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                return r0;
            }

            let tainted = infcx.region_vars.tainted(mark, r0);

            let mut a_r = None;
            let mut b_r = None;
            let mut only_new_vars = true;
            for r in tainted.iter() {
                if is_var_in_set(a_vars, *r) {
                    if a_r.is_some() {
                        return fresh_bound_variable(infcx, new_binder_id);
                    } else {
                        a_r = Some(*r);
                    }
                } else if is_var_in_set(b_vars, *r) {
                    if b_r.is_some() {
                        return fresh_bound_variable(infcx, new_binder_id);
                    } else {
                        b_r = Some(*r);
                    }
                } else if !is_var_in_set(new_vars, *r) {
                    only_new_vars = false;
                }
            }

            // NB---I do not believe this algorithm computes
            // (necessarily) the GLB.  As written it can
            // spuriously fail. In particular, if there is a case
            // like: |fn(&a)| and fn(fn(&b)), where a and b are
            // free, it will return fn(&c) where c = GLB(a,b).  If
            // however this GLB is not defined, then the result is
            // an error, even though something like
            // "fn<X>(fn(&X))" where X is bound would be a
            // subtype of both of those.
            //
            // The problem is that if we were to return a bound
            // variable, we'd be computing a lower-bound, but not
            // necessarily the *greatest* lower-bound.
            //
            // Unfortunately, this problem is non-trivial to solve,
            // because we do not know at the time of computing the GLB
            // whether a GLB(a,b) exists or not, because we haven't
            // run region inference (or indeed, even fully computed
            // the region hierarchy!). The current algorithm seems to
            // works ok in practice.

            if a_r.is_some() && b_r.is_some() && only_new_vars {
                // Related to exactly one bound variable from each fn:
                return rev_lookup(infcx, span, a_map, new_binder_id, a_r.unwrap());
            } else if a_r.is_none() && b_r.is_none() {
                // Not related to bound variables from either fn:
                assert!(!r0.is_bound());
                return r0;
            } else {
                // Other:
                return fresh_bound_variable(infcx, new_binder_id);
            }
        }

        fn rev_lookup(infcx: &InferCtxt,
                      span: Span,
                      a_map: &HashMap<ty::BoundRegion, ty::Region>,
                      new_binder_id: ast::NodeId,
                      r: ty::Region) -> ty::Region
        {
            for (a_br, a_r) in a_map.iter() {
                if *a_r == r {
                    return ty::ReLateBound(new_binder_id, *a_br);
                }
            }
            infcx.tcx.sess.span_bug(
                span,
                format!("could not find original bound region for {}", r)[]);
        }

        fn fresh_bound_variable(infcx: &InferCtxt, binder_id: ast::NodeId) -> ty::Region {
            infcx.region_vars.new_bound(binder_id)
        }
    }
}

impl HigherRanked for ty::FnSig {
    fn binder_id(&self) -> ast::NodeId {
        self.binder_id
    }

    fn super_combine<'tcx,C:Combine<'tcx>>(combine: &C, a: &ty::FnSig, b: &ty::FnSig)
                                           -> cres<ty::FnSig>
    {
        combine::super_fn_sigs(combine, a, b)
    }
}

pub fn var_ids<'tcx, T: Combine<'tcx>>(this: &T,
                                       map: &HashMap<ty::BoundRegion, ty::Region>)
                                       -> Vec<ty::RegionVid> {
    map.iter().map(|(_, r)| match *r {
            ty::ReInfer(ty::ReVar(r)) => { r }
            r => {
                this.infcx().tcx.sess.span_bug(
                    this.trace().origin.span(),
                    format!("found non-region-vid: {}", r).as_slice());
            }
        }).collect()
}

pub fn is_var_in_set(new_vars: &[ty::RegionVid], r: ty::Region) -> bool {
    match r {
        ty::ReInfer(ty::ReVar(ref v)) => new_vars.iter().any(|x| x == v),
        _ => false
    }
}

mod doc {
    /*!

# Skolemization and functions

One of the trickiest and most subtle aspects of regions is dealing
with higher-ranked things which include bound region variables, such
as function types. I strongly suggest that if you want to understand
the situation, you read this paper (which is, admittedly, very long,
but you don't have to read the whole thing):

http://research.microsoft.com/en-us/um/people/simonpj/papers/higher-rank/

Although my explanation will never compete with SPJ's (for one thing,
his is approximately 100 pages), I will attempt to explain the basic
problem and also how we solve it. Note that the paper only discusses
subtyping, not the computation of LUB/GLB.

The problem we are addressing is that there is a kind of subtyping
between functions with bound region parameters. Consider, for
example, whether the following relation holds:

    fn(&'a int) <: |&'b int|? (Yes, a => b)

The answer is that of course it does. These two types are basically
the same, except that in one we used the name `a` and one we used
the name `b`.

In the examples that follow, it becomes very important to know whether
a lifetime is bound in a function type (that is, is a lifetime
parameter) or appears free (is defined in some outer scope).
Therefore, from now on I will write the bindings explicitly, using a
notation like `fn<a>(&'a int)` to indicate that `a` is a lifetime
parameter.

Now let's consider two more function types. Here, we assume that the
`self` lifetime is defined somewhere outside and hence is not a
lifetime parameter bound by the function type (it "appears free"):

    fn<a>(&'a int) <: |&'a int|? (Yes, a => self)

This subtyping relation does in fact hold. To see why, you have to
consider what subtyping means. One way to look at `T1 <: T2` is to
say that it means that it is always ok to treat an instance of `T1` as
if it had the type `T2`. So, with our functions, it is always ok to
treat a function that can take pointers with any lifetime as if it
were a function that can only take a pointer with the specific
lifetime `&self`. After all, `&self` is a lifetime, after all, and
the function can take values of any lifetime.

You can also look at subtyping as the *is a* relationship. This amounts
to the same thing: a function that accepts pointers with any lifetime
*is a* function that accepts pointers with some specific lifetime.

So, what if we reverse the order of the two function types, like this:

    fn(&'a int) <: <a>|&'a int|? (No)

Does the subtyping relationship still hold?  The answer of course is
no. In this case, the function accepts *only the lifetime `&self`*,
so it is not reasonable to treat it as if it were a function that
accepted any lifetime.

What about these two examples:

    fn<a,b>(&'a int, &'b int) <: <a>|&'a int, &'a int|? (Yes)
    fn<a>(&'a int, &'a int) <: <a,b>|&'a int, &'b int|? (No)

Here, it is true that functions which take two pointers with any two
lifetimes can be treated as if they only accepted two pointers with
the same lifetime, but not the reverse.

## The algorithm

Here is the algorithm we use to perform the subtyping check:

1. Replace all bound regions in the subtype with new variables
2. Replace all bound regions in the supertype with skolemized
   equivalents. A "skolemized" region is just a new fresh region
   name.
3. Check that the parameter and return types match as normal
4. Ensure that no skolemized regions 'leak' into region variables
   visible from "the outside"

Let's walk through some examples and see how this algorithm plays out.

#### First example

We'll start with the first example, which was:

    1. fn<a>(&'a T) <: <b>|&'b T|?        Yes: a -> b

After steps 1 and 2 of the algorithm we will have replaced the types
like so:

    1. fn(&'A T) <: |&'x T|?

Here the upper case `&A` indicates a *region variable*, that is, a
region whose value is being inferred by the system. I also replaced
`&b` with `&x`---I'll use letters late in the alphabet (`x`, `y`, `z`)
to indicate skolemized region names. We can assume they don't appear
elsewhere. Note that neither the sub- nor the supertype bind any
region names anymore (as indicated by the absence of `<` and `>`).

The next step is to check that the parameter types match. Because
parameters are contravariant, this means that we check whether:

    &'x T <: &'A T

Region pointers are contravariant so this implies that

    &A <= &x

must hold, where `<=` is the subregion relationship. Processing
*this* constrain simply adds a constraint into our graph that `&A <=
&x` and is considered successful (it can, for example, be satisfied by
choosing the value `&x` for `&A`).

So far we have encountered no error, so the subtype check succeeds.

#### The third example

Now let's look first at the third example, which was:

    3. fn(&'a T)    <: <b>|&'b T|?        No!

After steps 1 and 2 of the algorithm we will have replaced the types
like so:

    3. fn(&'a T) <: |&'x T|?

This looks pretty much the same as before, except that on the LHS
`&self` was not bound, and hence was left as-is and not replaced with
a variable. The next step is again to check that the parameter types
match. This will ultimately require (as before) that `&self` <= `&x`
must hold: but this does not hold. `self` and `x` are both distinct
free regions. So the subtype check fails.

#### Checking for skolemization leaks

You may be wondering about that mysterious last step in the algorithm.
So far it has not been relevant. The purpose of that last step is to
catch something like *this*:

    fn<a>() -> fn(&'a T) <: || -> fn<b>(&'b T)?   No.

Here the function types are the same but for where the binding occurs.
The subtype returns a function that expects a value in precisely one
region. The supertype returns a function that expects a value in any
region. If we allow an instance of the subtype to be used where the
supertype is expected, then, someone could call the fn and think that
the return value has type `fn<b>(&'b T)` when it really has type
`fn(&'a T)` (this is case #3, above). Bad.

So let's step through what happens when we perform this subtype check.
We first replace the bound regions in the subtype (the supertype has
no bound regions). This gives us:

    fn() -> fn(&'A T) <: || -> fn<b>(&'b T)?

Now we compare the return types, which are covariant, and hence we have:

    fn(&'A T) <: <b>|&'b T|?

Here we skolemize the bound region in the supertype to yield:

    fn(&'A T) <: |&'x T|?

And then proceed to compare the argument types:

    &'x T <: &'A T
    &A <= &x

Finally, this is where it gets interesting!  This is where an error
*should* be reported. But in fact this will not happen. The reason why
is that `A` is a variable: we will infer that its value is the fresh
region `x` and think that everything is happy. In fact, this behavior
is *necessary*, it was key to the first example we walked through.

The difference between this example and the first one is that the variable
`A` already existed at the point where the skolemization occurred. In
the first example, you had two functions:

    fn<a>(&'a T) <: <b>|&'b T|

and hence `&A` and `&x` were created "together". In general, the
intention of the skolemized names is that they are supposed to be
fresh names that could never be equal to anything from the outside.
But when inference comes into play, we might not be respecting this
rule.

So the way we solve this is to add a fourth step that examines the
constraints that refer to skolemized names. Basically, consider a
non-directed verison of the constraint graph. Let `Tainted(x)` be the
set of all things reachable from a skolemized variable `x`.
`Tainted(x)` should not contain any regions that existed before the
step at which the skolemization was performed. So this case here
would fail because `&x` was created alone, but is relatable to `&A`.

## Computing the LUB and GLB

The paper I pointed you at is written for Haskell. It does not
therefore considering subtyping and in particular does not consider
LUB or GLB computation. We have to consider this. Here is the
algorithm I implemented.

First though, let's discuss what we are trying to compute in more
detail. The LUB is basically the "common supertype" and the GLB is
"common subtype"; one catch is that the LUB should be the
*most-specific* common supertype and the GLB should be *most general*
common subtype (as opposed to any common supertype or any common
subtype).

Anyway, to help clarify, here is a table containing some
function pairs and their LUB/GLB:

```
Type 1              Type 2              LUB               GLB
fn<a>(&a)           fn(&X)              fn(&X)            fn<a>(&a)
fn(&A)              fn(&X)              --                fn<a>(&a)
fn<a,b>(&a, &b)     fn<x>(&x, &x)       fn<a>(&a, &a)     fn<a,b>(&a, &b)
fn<a,b>(&a, &b, &a) fn<x,y>(&x, &y, &y) fn<a>(&a, &a, &a) fn<a,b,c>(&a,&b,&c)
```

     */
}
