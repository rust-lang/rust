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

use core::prelude::*;

pub use middle::ty::IntVarValue;
pub use middle::typeck::infer::resolve::resolve_and_force_all_but_regions;
pub use middle::typeck::infer::resolve::{force_all, not_regions};
pub use middle::typeck::infer::resolve::{force_ivar};
pub use middle::typeck::infer::resolve::{force_tvar, force_rvar};
pub use middle::typeck::infer::resolve::{resolve_ivar, resolve_all};
pub use middle::typeck::infer::resolve::{resolve_nested_tvar};
pub use middle::typeck::infer::resolve::{resolve_rvar};

use middle::ty::{TyVid, IntVid, FloatVid, RegionVid, Vid};
use middle::ty;
use middle::typeck::check::regionmanip::{replace_bound_regions_in_fn_sig};
use middle::typeck::infer::coercion::Coerce;
use middle::typeck::infer::combine::{Combine, CombineFields, eq_tys};
use middle::typeck::infer::region_inference::{RegionVarBindings};
use middle::typeck::infer::resolve::{resolver};
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::unify::{ValsAndBindings, Root};
use middle::typeck::isr_alist;
use util::common::indent;
use util::ppaux::{bound_region_to_str, ty_to_str, trait_ref_to_str};

use core::cmp::Eq;
use core::result::{Result, Ok, Err};
use core::result;
use core::vec;
use std::list::Nil;
use std::smallintmap::SmallIntMap;
use syntax::ast::{m_imm, m_mutbl};
use syntax::ast;
use syntax::codemap;
use syntax::codemap::span;

pub mod macros;
pub mod combine;
pub mod glb;
pub mod lattice;
pub mod lub;
pub mod region_inference;
pub mod resolve;
pub mod sub;
pub mod to_str;
pub mod unify;
pub mod coercion;

pub type Bound<T> = Option<T>;
pub struct Bounds<T> {
    lb: Bound<T>,
    ub: Bound<T>
}

pub type cres<T> = Result<T,ty::type_err>; // "combine result"
pub type ures = cres<()>; // "unify result"
pub type fres<T> = Result<T, fixup_err>; // "fixup result"
pub type CoerceResult = cres<Option<@ty::AutoAdjustment>>;

pub struct InferCtxt {
    tcx: ty::ctxt,

    // We instantiate ValsAndBindings with bounds<ty::t> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    ty_var_bindings: ValsAndBindings<ty::TyVid, Bounds<ty::t>>,
    ty_var_counter: uint,

    // Map from integral variable to the kind of integer it represents
    int_var_bindings: ValsAndBindings<ty::IntVid, Option<IntVarValue>>,
    int_var_counter: uint,

    // Map from floating variable to the kind of float it represents
    float_var_bindings: ValsAndBindings<ty::FloatVid, Option<ast::float_ty>>,
    float_var_counter: uint,

    // For region variables.
    region_vars: RegionVarBindings,
}

pub enum fixup_err {
    unresolved_int_ty(IntVid),
    unresolved_ty(TyVid),
    cyclic_ty(TyVid),
    unresolved_region(RegionVid),
    region_var_bound_by_region_var(RegionVid, RegionVid)
}

pub fn fixup_err_to_str(f: fixup_err) -> ~str {
    match f {
      unresolved_int_ty(_) => ~"unconstrained integral type",
      unresolved_ty(_) => ~"unconstrained type",
      cyclic_ty(_) => ~"cyclic type of infinite size",
      unresolved_region(_) => ~"unconstrained region",
      region_var_bound_by_region_var(r1, r2) => {
        fmt!("region var %? bound by another region var %?; this is \
              a bug in rustc", r1, r2)
      }
    }
}

fn new_ValsAndBindings<V:Copy,T:Copy>() -> ValsAndBindings<V, T> {
    ValsAndBindings {
        vals: @mut SmallIntMap::new(),
        bindings: ~[]
    }
}

pub fn new_infer_ctxt(tcx: ty::ctxt) -> @mut InferCtxt {
    @mut InferCtxt {
        tcx: tcx,

        ty_var_bindings: new_ValsAndBindings(),
        ty_var_counter: 0,

        int_var_bindings: new_ValsAndBindings(),
        int_var_counter: 0,

        float_var_bindings: new_ValsAndBindings(),
        float_var_counter: 0,

        region_vars: RegionVarBindings(tcx),
    }
}

pub fn mk_subty(cx: @mut InferCtxt,
                a_is_expected: bool,
                span: span,
                a: ty::t,
                b: ty::t)
             -> ures {
    debug!("mk_subty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            cx.sub(a_is_expected, span).tys(a, b)
        }
    }.to_ures()
}

pub fn can_mk_subty(cx: @mut InferCtxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_subty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.probe {
            cx.sub(true, codemap::dummy_sp()).tys(a, b)
        }
    }.to_ures()
}

pub fn mk_subr(cx: @mut InferCtxt,
               a_is_expected: bool,
               span: span,
               a: ty::Region,
               b: ty::Region)
            -> ures {
    debug!("mk_subr(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            cx.sub(a_is_expected, span).regions(a, b)
        }
    }.to_ures()
}

pub fn mk_eqty(cx: @mut InferCtxt,
               a_is_expected: bool,
               span: span,
               a: ty::t,
               b: ty::t)
            -> ures {
    debug!("mk_eqty(%s <: %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let suber = cx.sub(a_is_expected, span);
            eq_tys(&suber, a, b)
        }
    }.to_ures()
}

pub fn mk_sub_trait_refs(cx: @mut InferCtxt,
                         a_is_expected: bool,
                         span: span,
                         a: &ty::TraitRef,
                         b: &ty::TraitRef)
    -> ures
{
    debug!("mk_sub_trait_refs(%s <: %s)",
           a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            let suber = cx.sub(a_is_expected, span);
            suber.trait_refs(a, b)
        }
    }.to_ures()
}

pub fn mk_coercety(cx: @mut InferCtxt,
                   a_is_expected: bool,
                   span: span,
                   a: ty::t,
                   b: ty::t)
                -> CoerceResult {
    debug!("mk_coercety(%s -> %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.commit {
            Coerce(cx.combine_fields(a_is_expected, span)).tys(a, b)
        }
    }
}

pub fn can_mk_coercety(cx: @mut InferCtxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_coercety(%s -> %s)", a.inf_str(cx), b.inf_str(cx));
    do indent {
        do cx.probe {
            let span = codemap::dummy_sp();
            Coerce(cx.combine_fields(true, span)).tys(a, b)
        }
    }.to_ures()
}

// See comment on the type `resolve_state` below
pub fn resolve_type(cx: @mut InferCtxt,
                    a: ty::t,
                    modes: uint)
                 -> fres<ty::t> {
    let mut resolver = resolver(cx, modes);
    resolver.resolve_type_chk(a)
}

pub fn resolve_region(cx: @mut InferCtxt, r: ty::Region, modes: uint)
                   -> fres<ty::Region> {
    let mut resolver = resolver(cx, modes);
    resolver.resolve_region_chk(r)
}

/*
fn resolve_borrowings(cx: @mut InferCtxt) {
    for cx.borrowings.each |item| {
        match resolve_region(cx, item.scope, resolve_all|force_all) {
          Ok(region) => {
            debug!("borrowing for expr %d resolved to region %?, mutbl %?",
                   item.expr_id, region, item.mutbl);
            cx.tcx.borrowings.insert(
                item.expr_id, {region: region, mutbl: item.mutbl});
          }

          Err(e) => {
            let str = fixup_err_to_str(e);
            cx.tcx.sess.span_err(
                item.span,
                fmt!("could not resolve lifetime for borrow: %s", str));
          }
        }
    }
}
*/

trait then {
    fn then<T:Copy>(&self, f: &fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err>;
}

impl then for ures {
    fn then<T:Copy>(&self, f: &fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err> {
        self.chain(|_i| f())
    }
}

trait ToUres {
    fn to_ures(&self) -> ures;
}

impl<T> ToUres for cres<T> {
    fn to_ures(&self) -> ures {
        match *self {
          Ok(ref _v) => Ok(()),
          Err(ref e) => Err((*e))
        }
    }
}

trait CresCompare<T> {
    fn compare(&self, t: T, f: &fn() -> ty::type_err) -> cres<T>;
}

impl<T:Copy + Eq> CresCompare<T> for cres<T> {
    fn compare(&self, t: T, f: &fn() -> ty::type_err) -> cres<T> {
        do self.chain |s| {
            if s == t {
                *self
            } else {
                Err(f())
            }
        }
    }
}

pub fn uok() -> ures {
    Ok(())
}

fn rollback_to<V:Copy + Vid,T:Copy>(
    vb: &mut ValsAndBindings<V, T>,
    len: uint)
{
    while vb.bindings.len() != len {
        let (vid, old_v) = vb.bindings.pop();
        vb.vals.insert(vid.to_uint(), old_v);
    }
}

struct Snapshot {
    ty_var_bindings_len: uint,
    int_var_bindings_len: uint,
    float_var_bindings_len: uint,
    region_vars_snapshot: uint,
}

pub impl InferCtxt {
    fn combine_fields(@mut self, a_is_expected: bool,
                      span: span) -> CombineFields {
        CombineFields {infcx: self,
                       a_is_expected: a_is_expected,
                       span: span}
    }

    fn sub(@mut self, a_is_expected: bool, span: span) -> Sub {
        Sub(self.combine_fields(a_is_expected, span))
    }

    fn in_snapshot(@mut self) -> bool {
        self.region_vars.in_snapshot()
    }

    fn start_snapshot(@mut self) -> Snapshot {
        let this = &mut *self;
        Snapshot {
            ty_var_bindings_len:
                this.ty_var_bindings.bindings.len(),
            int_var_bindings_len:
                this.int_var_bindings.bindings.len(),
            float_var_bindings_len:
                this.float_var_bindings.bindings.len(),
            region_vars_snapshot:
                this.region_vars.start_snapshot(),
        }
    }

    fn rollback_to(@mut self, snapshot: &Snapshot) {
        debug!("rollback!");
        rollback_to(&mut self.ty_var_bindings, snapshot.ty_var_bindings_len);

        rollback_to(&mut self.int_var_bindings,
                    snapshot.int_var_bindings_len);
        rollback_to(&mut self.float_var_bindings,
                    snapshot.float_var_bindings_len);

        self.region_vars.rollback_to(snapshot.region_vars_snapshot);
    }

    /// Execute `f` and commit the bindings if successful
    fn commit<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        assert!(!self.in_snapshot());

        debug!("commit()");
        do indent {
            let r = self.try(f);

            self.ty_var_bindings.bindings.truncate(0);
            self.int_var_bindings.bindings.truncate(0);
            self.region_vars.commit();
            r
        }
    }

    /// Execute `f`, unroll bindings on failure
    fn try<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        debug!("try()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = f();
            match r {
              Ok(_) => (),
              Err(_) => self.rollback_to(&snapshot)
            }
            r
        }
    }

    /// Execute `f` then unroll any bindings it creates
    fn probe<T,E>(@mut self, f: &fn() -> Result<T,E>) -> Result<T,E> {
        debug!("probe()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = self.try(f);
            self.rollback_to(&snapshot);
            r
        }
    }
}

fn next_simple_var<V:Copy,T:Copy>(
        +counter: &mut uint,
        +bindings: &mut ValsAndBindings<V,Option<T>>)
     -> uint {
    let id = *counter;
    *counter += 1;
    bindings.vals.insert(id, Root(None, 0));
    return id;
}

pub impl InferCtxt {
    fn next_ty_var_id(@mut self) -> TyVid {
        let id = self.ty_var_counter;
        self.ty_var_counter += 1;
        let vals = self.ty_var_bindings.vals;
        vals.insert(id, Root(Bounds { lb: None, ub: None }, 0u));
        return TyVid(id);
    }

    fn next_ty_var(@mut self) -> ty::t {
        ty::mk_var(self.tcx, self.next_ty_var_id())
    }

    fn next_ty_vars(@mut self, n: uint) -> ~[ty::t] {
        vec::from_fn(n, |_i| self.next_ty_var())
    }

    fn next_int_var_id(@mut self) -> IntVid {
        IntVid(next_simple_var(&mut self.int_var_counter,
                               &mut self.int_var_bindings))
    }

    fn next_int_var(@mut self) -> ty::t {
        ty::mk_int_var(self.tcx, self.next_int_var_id())
    }

    fn next_float_var_id(@mut self) -> FloatVid {
        FloatVid(next_simple_var(&mut self.float_var_counter,
                                 &mut self.float_var_bindings))
    }

    fn next_float_var(@mut self) -> ty::t {
        ty::mk_float_var(self.tcx, self.next_float_var_id())
    }

    fn next_region_var_nb(@mut self, span: span) -> ty::Region {
        ty::re_infer(ty::ReVar(self.region_vars.new_region_var(span)))
    }

    fn next_region_var_with_lb(@mut self, span: span,
                               lb_region: ty::Region) -> ty::Region {
        let region_var = self.next_region_var_nb(span);

        // add lb_region as a lower bound on the newly built variable
        assert!(self.region_vars.make_subregion(span,
                                                     lb_region,
                                                     region_var).is_ok());

        return region_var;
    }

    fn next_region_var(@mut self, span: span, scope_id: ast::node_id)
                      -> ty::Region {
        self.next_region_var_with_lb(span, ty::re_scope(scope_id))
    }

    fn resolve_regions(@mut self) {
        self.region_vars.resolve_regions();
    }

    fn ty_to_str(@mut self, t: ty::t) -> ~str {
        ty_to_str(self.tcx,
                  self.resolve_type_vars_if_possible(t))
    }

    fn trait_ref_to_str(@mut self, t: &ty::TraitRef) -> ~str {
        let t = self.resolve_type_vars_in_trait_ref_if_possible(t);
        trait_ref_to_str(self.tcx, &t)
    }

    fn resolve_type_vars_if_possible(@mut self, typ: ty::t) -> ty::t {
        match resolve_type(self, typ, resolve_nested_tvar | resolve_ivar) {
          result::Ok(new_type) => new_type,
          result::Err(_) => typ
        }
    }

    fn resolve_type_vars_in_trait_ref_if_possible(@mut self,
                                                  trait_ref: &ty::TraitRef)
        -> ty::TraitRef
    {
        // make up a dummy type just to reuse/abuse the resolve machinery
        let dummy0 = ty::mk_trait(self.tcx,
                                  trait_ref.def_id,
                                  copy trait_ref.substs,
                                  ty::UniqTraitStore);
        let dummy1 = self.resolve_type_vars_if_possible(dummy0);
        match ty::get(dummy1).sty {
            ty::ty_trait(ref def_id, ref substs, _) => {
                ty::TraitRef {def_id: *def_id,
                              substs: copy *substs}
            }
            _ => {
                self.tcx.sess.bug(
                    fmt!("resolve_type_vars_if_possible() yielded %s \
                          when supplied with %s",
                         self.ty_to_str(dummy0),
                         self.ty_to_str(dummy1)));
            }
        }
    }

    fn type_error_message(@mut self, sp: span, mk_msg: &fn(~str) -> ~str,
                          actual_ty: ty::t, err: Option<&ty::type_err>) {
        let actual_ty = self.resolve_type_vars_if_possible(actual_ty);

        // Don't report an error if actual type is ty_err.
        if ty::type_is_error(actual_ty) {
            return;
        }
        let error_str = err.map_default(~"", |t_err|
                         fmt!(" (%s)",
                              ty::type_err_to_str(self.tcx, *t_err)));
        self.tcx.sess.span_err(sp,
           fmt!("%s%s", mk_msg(self.ty_to_str(actual_ty)),
                error_str));
        for err.each |err| {
            ty::note_and_explain_type_err(self.tcx, *err)
        }
    }

    fn report_mismatched_types(@mut self, sp: span, e: ty::t, a: ty::t,
                               err: &ty::type_err) {
        let resolved_expected =
            self.resolve_type_vars_if_possible(e);
        let mk_msg = match ty::get(resolved_expected).sty {
            // Don't report an error if expected is ty_err
            ty::ty_err => return,
            _ => {
                // if I leave out : ~str, it infers &str and complains
                |actual: ~str| {
                    fmt!("mismatched types: expected `%s` but found `%s`",
                         self.ty_to_str(resolved_expected), actual)
                }
            }
        };
        self.type_error_message(sp, mk_msg, a, Some(err));
    }

    fn replace_bound_regions_with_fresh_regions(@mut self,
            span: span,
            fsig: &ty::FnSig)
         -> (ty::FnSig, isr_alist) {
        let(isr, _, fn_sig) =
            replace_bound_regions_in_fn_sig(self.tcx, @Nil, None, fsig, |br| {
                // N.B.: The name of the bound region doesn't have anything to
                // do with the region variable that's created for it.  The
                // only thing we're doing with `br` here is using it in the
                // debug message.
                let rvar = self.next_region_var_nb(span);
                debug!("Bound region %s maps to %?",
                       bound_region_to_str(self.tcx, br),
                       rvar);
                rvar
            });
        (fn_sig, isr)
    }

    fn fold_regions_in_sig(
        @mut self,
        fn_sig: &ty::FnSig,
        fldr: &fn(r: ty::Region, in_fn: bool) -> ty::Region) -> ty::FnSig
    {
        do ty::fold_sig(fn_sig) |t| {
            ty::fold_regions(self.tcx, t, fldr)
        }
    }

}
