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

## Integral type variables

There is a third variety of type variable that we use only for
inferring the types of unsuffixed integer literals.  Integral type
variables differ from general-purpose type variables in that there's
no subtyping relationship among the various integral types, so instead
of associating each variable with an upper and lower bound, we
represent the set of possible integral types it can take on with an
`int_ty_set`, which is a bitvector with one bit for each integral
type.  Because intersecting these sets with each other is simpler than
merging bounds, we don't need to do so transactionally as we do for
general-purpose type variables.

We could conceivably define a subtyping relationship among integral
types based on their ranges, but we choose not to open that particular
can of worms.  Our strategy is to treat integral type variables as
unknown until the typing context constrains them to a unique integral
type, at which point they take on that type.  If the typing context
overconstrains the type, it's a type error; if we reach the point at
which type variables must be resolved and an integral type variable is
still underconstrained, it defaults to `int` as a last resort.

Floating point types are handled similarly to integral types.

## GLB/LUB

Computing the greatest-lower-bound and least-upper-bound of two
types/regions is generally straightforward except when type variables
are involved. In that case, we follow a similar "try to use the bounds
when possible but otherwise merge the variables" strategy.  In other
words, `GLB(A, B)` where `A` and `B` are variables will often result
in `A` and `B` being merged and the result being `A`.

## Type assignment

We have a notion of assignability which differs somewhat from
subtyping; in particular it may cause region borrowing to occur.  See
the big comment later in this file on Type Assignment for specifics.

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

#[legacy_exports];
#[warn(deprecated_mode)];
#[warn(deprecated_pattern)];

use std::smallintmap;
use std::smallintmap::smallintmap;
use std::map::HashMap;
use middle::ty;
use middle::ty::{TyVid, IntVid, FloatVid, RegionVid, vid,
                 ty_int, ty_uint, get, terr_fn, TyVar, IntVar, FloatVar};
use syntax::{ast, ast_util};
use syntax::ast::{ret_style, purity};
use util::ppaux::{ty_to_str, mt_to_str};
use result::{Result, Ok, Err, map_vec, map_vec2, iter_vec2};
use ty::{mk_fn, type_is_bot};
use check::regionmanip::{replace_bound_regions_in_fn_ty};
use util::common::{indent, indenter};
use ast::{unsafe_fn, impure_fn, pure_fn, extern_fn};
use ast::{m_const, m_imm, m_mutbl};
use dvec::DVec;
use region_inference::{RegionVarBindings};
use ast_util::dummy_sp;
use cmp::Eq;

// From submodules:
use resolve::{resolve_nested_tvar, resolve_rvar, resolve_ivar, resolve_all,
                 force_tvar, force_rvar, force_ivar, force_all, not_regions,
                 resolve_and_force_all_but_regions, resolver};
use unify::{vals_and_bindings, root};
use integral::{int_ty_set, int_ty_set_all};
use floating::{float_ty_set, float_ty_set_all};
use combine::{combine_fields, eq_tys};
use assignment::Assign;
use to_str::ToStr;

use sub::Sub;
use lub::Lub;
use glb::Glb;

export infer_ctxt;
export new_infer_ctxt;
export mk_subty, can_mk_subty;
export mk_subr;
export mk_eqty;
export mk_assignty, can_mk_assignty;
export resolve_nested_tvar, resolve_rvar, resolve_ivar, resolve_all;
export force_tvar, force_rvar, force_ivar, force_all;
export resolve_and_force_all_but_regions, not_regions;
export resolve_type, resolve_region;
export resolve_borrowings;
export methods; // for infer_ctxt
export unify_methods; // for infer_ctxt
export cres, fres, fixup_err, fixup_err_to_str;
export assignment;
export root, to_str;
export int_ty_set_all;

#[legacy_exports]
mod assignment;
#[legacy_exports]
mod combine;
#[legacy_exports]
mod glb;
#[legacy_exports]
mod integral;
mod floating;
#[legacy_exports]
mod lattice;
#[legacy_exports]
mod lub;
#[legacy_exports]
mod region_inference;
#[legacy_exports]
mod resolve;
#[legacy_exports]
mod sub;
#[legacy_exports]
mod to_str;
#[legacy_exports]
mod unify;

type bound<T:Copy> = Option<T>;
type bounds<T:Copy> = {lb: bound<T>, ub: bound<T>};

type cres<T> = Result<T,ty::type_err>; // "combine result"
type ures = cres<()>; // "unify result"
type fres<T> = Result<T, fixup_err>; // "fixup result"
type ares = cres<Option<@ty::AutoAdjustment>>; // "assignment result"

enum infer_ctxt = @{
    tcx: ty::ctxt,

    // We instantiate vals_and_bindings with bounds<ty::t> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    ty_var_bindings: vals_and_bindings<ty::TyVid, bounds<ty::t>>,

    // The types that might instantiate an integral type variable are
    // represented by an int_ty_set.
    int_var_bindings: vals_and_bindings<ty::IntVid, int_ty_set>,

    // The types that might instantiate a floating-point type variable are
    // represented by an float_ty_set.
    float_var_bindings: vals_and_bindings<ty::FloatVid, float_ty_set>,

    // For region variables.
    region_vars: RegionVarBindings,

    // For keeping track of existing type and region variables.
    ty_var_counter: @mut uint,
    int_var_counter: @mut uint,
    float_var_counter: @mut uint,
    region_var_counter: @mut uint
};

enum fixup_err {
    unresolved_int_ty(IntVid),
    unresolved_ty(TyVid),
    cyclic_ty(TyVid),
    unresolved_region(RegionVid),
    region_var_bound_by_region_var(RegionVid, RegionVid)
}

fn fixup_err_to_str(f: fixup_err) -> ~str {
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

fn new_vals_and_bindings<V:Copy, T:Copy>() -> vals_and_bindings<V, T> {
    vals_and_bindings {
        vals: smallintmap::mk(),
        mut bindings: ~[]
    }
}

fn new_infer_ctxt(tcx: ty::ctxt) -> infer_ctxt {
    infer_ctxt(@{tcx: tcx,
                 ty_var_bindings: new_vals_and_bindings(),
                 int_var_bindings: new_vals_and_bindings(),
                 float_var_bindings: new_vals_and_bindings(),
                 region_vars: RegionVarBindings(tcx),
                 ty_var_counter: @mut 0u,
                 int_var_counter: @mut 0u,
                 float_var_counter: @mut 0u,
                 region_var_counter: @mut 0u})}

fn mk_subty(cx: infer_ctxt, a_is_expected: bool, span: span,
            a: ty::t, b: ty::t) -> ures {
    debug!("mk_subty(%s <: %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.commit {
            cx.sub(a_is_expected, span).tys(a, b)
        }
    }.to_ures()
}

fn can_mk_subty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_subty(%s <: %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.probe {
            cx.sub(true, ast_util::dummy_sp()).tys(a, b)
        }
    }.to_ures()
}

fn mk_subr(cx: infer_ctxt, a_is_expected: bool, span: span,
           a: ty::Region, b: ty::Region) -> ures {
    debug!("mk_subr(%s <: %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.commit {
            cx.sub(a_is_expected, span).regions(a, b)
        }
    }.to_ures()
}

fn mk_eqty(cx: infer_ctxt, a_is_expected: bool, span: span,
           a: ty::t, b: ty::t) -> ures {
    debug!("mk_eqty(%s <: %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.commit {
            let suber = cx.sub(a_is_expected, span);
            eq_tys(&suber, a, b)
        }
    }.to_ures()
}

fn mk_assignty(cx: infer_ctxt, a_is_expected: bool, span: span,
               a: ty::t, b: ty::t) -> ares {
    debug!("mk_assignty(%s -> %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.commit {
            Assign(cx.combine_fields(a_is_expected, span)).tys(a, b)
        }
    }
}

fn can_mk_assignty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    debug!("can_mk_assignty(%s -> %s)", a.to_str(cx), b.to_str(cx));
    do indent {
        do cx.probe {
            let span = ast_util::dummy_sp();
            Assign(cx.combine_fields(true, span)).tys(a, b)
        }
    }.to_ures()
}

// See comment on the type `resolve_state` below
fn resolve_type(cx: infer_ctxt, a: ty::t, modes: uint)
    -> fres<ty::t> {
    resolver(cx, modes).resolve_type_chk(a)
}

fn resolve_region(cx: infer_ctxt, r: ty::Region, modes: uint)
    -> fres<ty::Region> {
    resolver(cx, modes).resolve_region_chk(r)
}

/*
fn resolve_borrowings(cx: infer_ctxt) {
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
    fn then<T:Copy>(f: fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err>;
}

impl ures: then {
    fn then<T:Copy>(f: fn() -> Result<T,ty::type_err>)
        -> Result<T,ty::type_err> {
        self.chain(|_i| f())
    }
}

trait ToUres {
    fn to_ures() -> ures;
}

impl<T> cres<T>: ToUres {
    fn to_ures() -> ures {
        match self {
          Ok(_v) => Ok(()),
          Err(e) => Err(e)
        }
    }
}

trait CresCompare<T> {
    fn compare(t: T, f: fn() -> ty::type_err) -> cres<T>;
}

impl<T:Copy Eq> cres<T>: CresCompare<T> {
    fn compare(t: T, f: fn() -> ty::type_err) -> cres<T> {
        do self.chain |s| {
            if s == t {
                self
            } else {
                Err(f())
            }
        }
    }
}

fn uok() -> ures {
    Ok(())
}

fn rollback_to<V:Copy vid, T:Copy>(
    vb: &vals_and_bindings<V, T>, len: uint) {

    while vb.bindings.len() != len {
        let (vid, old_v) = vb.bindings.pop();
        vb.vals.insert(vid.to_uint(), old_v);
    }
}

struct Snapshot {
    ty_var_bindings_len: uint,
    int_var_bindings_len: uint,
    region_vars_snapshot: uint,
}

impl infer_ctxt {
    fn combine_fields(a_is_expected: bool,
                      span: span) -> combine_fields {
        combine_fields {infcx: self,
                        a_is_expected: a_is_expected,
                        span: span}
    }

    fn sub(a_is_expected: bool, span: span) -> Sub {
        Sub(self.combine_fields(a_is_expected, span))
    }

    fn in_snapshot() -> bool {
        self.region_vars.in_snapshot()
    }

    fn start_snapshot() -> Snapshot {
        Snapshot {
            ty_var_bindings_len:
                self.ty_var_bindings.bindings.len(),
            int_var_bindings_len:
                self.int_var_bindings.bindings.len(),
            region_vars_snapshot:
                self.region_vars.start_snapshot(),
        }
    }

    fn rollback_to(snapshot: &Snapshot) {
        debug!("rollback!");
        rollback_to(&self.ty_var_bindings, snapshot.ty_var_bindings_len);

        // FIXME(#3211) -- int_var not transactional
        //rollback_to(&self.int_var_bindings,
        //            snapshot.int_var_bindings_len);

        self.region_vars.rollback_to(
            snapshot.region_vars_snapshot);
    }

    /// Execute `f` and commit the bindings if successful
    fn commit<T,E>(f: fn() -> Result<T,E>) -> Result<T,E> {
        assert !self.in_snapshot();

        debug!("commit()");
        do indent {
            let r = self.try(f);

            self.ty_var_bindings.bindings.truncate(0);
            self.int_var_bindings.bindings.truncate(0);
            self.region_vars.commit();
            move r
        }
    }

    /// Execute `f`, unroll bindings on failure
    fn try<T,E>(f: fn() -> Result<T,E>) -> Result<T,E> {
        debug!("try()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = f();
            match r {
              Ok(_) => (),
              Err(_) => self.rollback_to(&snapshot)
            }
            move r
        }
    }

    /// Execute `f` then unroll any bindings it creates
    fn probe<T,E>(f: fn() -> Result<T,E>) -> Result<T,E> {
        debug!("probe()");
        do indent {
            let snapshot = self.start_snapshot();
            let r = self.try(f);
            self.rollback_to(&snapshot);
            move r
        }
    }
}

impl infer_ctxt {
    fn next_ty_var_id() -> TyVid {
        let id = *self.ty_var_counter;
        *self.ty_var_counter += 1u;
        self.ty_var_bindings.vals.insert(id,
                                         root({lb: None, ub: None}, 0u));
        return TyVid(id);
    }

    fn next_ty_var() -> ty::t {
        ty::mk_var(self.tcx, self.next_ty_var_id())
    }

    fn next_ty_vars(n: uint) -> ~[ty::t] {
        vec::from_fn(n, |_i| self.next_ty_var())
    }

    fn next_int_var_id() -> IntVid {
        let id = *self.int_var_counter;
        *self.int_var_counter += 1u;

        self.int_var_bindings.vals.insert(id,
                              root(int_ty_set_all(), 0u));
        return IntVid(id);
    }

    fn next_int_var() -> ty::t {
        ty::mk_int_var(self.tcx, self.next_int_var_id())
    }

    fn next_float_var_id() -> FloatVid {
        let id = *self.float_var_counter;
        *self.float_var_counter += 1;

        self.float_var_bindings.vals.insert(id, root(float_ty_set_all(), 0));
        return FloatVid(id);
    }

    fn next_float_var() -> ty::t {
        ty::mk_float_var(self.tcx, self.next_float_var_id())
    }

    fn next_region_var_nb(span: span) -> ty::Region {
        ty::re_infer(ty::ReVar(self.region_vars.new_region_var(span)))
    }

    fn next_region_var_with_lb(span: span,
                               lb_region: ty::Region) -> ty::Region {
        let region_var = self.next_region_var_nb(span);

        // add lb_region as a lower bound on the newly built variable
        assert self.region_vars.make_subregion(span,
                                               lb_region,
                                               region_var).is_ok();

        return region_var;
    }

    fn next_region_var(span: span, scope_id: ast::node_id) -> ty::Region {
        self.next_region_var_with_lb(span, ty::re_scope(scope_id))
    }

    fn resolve_regions() {
        self.region_vars.resolve_regions();
    }

    fn ty_to_str(t: ty::t) -> ~str {
        ty_to_str(self.tcx,
                  self.resolve_type_vars_if_possible(t))
    }

    fn resolve_type_vars_if_possible(typ: ty::t) -> ty::t {
        match resolve_type(self, typ, resolve_nested_tvar | resolve_ivar) {
          result::Ok(new_type) => new_type,
          result::Err(_) => typ
        }
    }

    fn type_error_message(sp: span, mk_msg: fn(~str) -> ~str,
                          actual_ty: ty::t, err: Option<&ty::type_err>) {
        let actual_ty = self.resolve_type_vars_if_possible(actual_ty);

        // Don't report an error if actual type is ty_err.
        match ty::get(actual_ty).sty {
            ty::ty_err => return,
            _           => ()
        }
        let error_str = err.map_default(~"", |t_err|
                         fmt!(" (%s)",
                              ty::type_err_to_str(self.tcx, *t_err)));
        self.tcx.sess.span_err(sp,
           fmt!("%s%s", mk_msg(self.ty_to_str(actual_ty)),
                error_str));
        err.iter(|err|
             ty::note_and_explain_type_err(self.tcx, *err));
    }

    fn report_mismatched_types(sp: span, e: ty::t, a: ty::t,
                               err: &ty::type_err) {
        // Don't report an error if expected is ty_err
        let resolved_expected =
            self.resolve_type_vars_if_possible(e);
        let mk_msg = match ty::get(resolved_expected).sty {
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

    fn replace_bound_regions_with_fresh_regions(
        &self, span: span,
        fty: &ty::FnTy) -> (ty::FnTy, isr_alist)
    {
        let {fn_ty, isr, _} =
            replace_bound_regions_in_fn_ty(self.tcx, @Nil, None, fty, |br| {
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
        (fn_ty, isr)
    }

}

