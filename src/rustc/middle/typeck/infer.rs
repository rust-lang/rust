/*

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

#[warn(deprecated_mode)];
#[warn(deprecated_pattern)];

import std::smallintmap;
import std::smallintmap::smallintmap;
import std::map::hashmap;
import middle::ty;
import middle::ty::{tv_vid, tvi_vid, region_vid, vid,
                    ty_int, ty_uint, get, terr_fn};
import syntax::{ast, ast_util};
import syntax::ast::{ret_style, purity};
import util::ppaux::{ty_to_str, mt_to_str};
import result::{result, ok, err, map_vec, map_vec2, iter_vec2};
import ty::{mk_fn, type_is_bot};
import check::regionmanip::{replace_bound_regions_in_fn_ty};
import driver::session::session;
import util::common::{indent, indenter};
import ast::{unsafe_fn, impure_fn, pure_fn, extern_fn};
import ast::{m_const, m_imm, m_mutbl};
import dvec::{DVec, dvec};

export infer_ctxt;
export new_infer_ctxt;
export mk_subty, can_mk_subty;
export mk_subr;
export mk_eqty;
export mk_assignty, can_mk_assignty;
export resolve_nested_tvar, resolve_rvar, resolve_ivar, resolve_all;
export force_tvar, force_rvar, force_ivar, force_all;
export resolve_type, resolve_region;
export resolve_borrowings;
export methods; // for infer_ctxt
export unify_methods; // for infer_ctxt
export fres, fixup_err, fixup_err_to_str;
export assignment;
export root, to_str;
export int_ty_set_all;

// Bitvector to represent sets of integral types
enum int_ty_set = uint;

// Constants representing singleton sets containing each of the
// integral types
const INT_TY_SET_EMPTY : uint = 0b00_0000_0000u;
const INT_TY_SET_i8    : uint = 0b00_0000_0001u;
const INT_TY_SET_u8    : uint = 0b00_0000_0010u;
const INT_TY_SET_i16   : uint = 0b00_0000_0100u;
const INT_TY_SET_u16   : uint = 0b00_0000_1000u;
const INT_TY_SET_i32   : uint = 0b00_0001_0000u;
const INT_TY_SET_u32   : uint = 0b00_0010_0000u;
const INT_TY_SET_i64   : uint = 0b00_0100_0000u;
const INT_TY_SET_u64   : uint = 0b00_1000_0000u;
const INT_TY_SET_i     : uint = 0b01_0000_0000u;
const INT_TY_SET_u     : uint = 0b10_0000_0000u;

fn int_ty_set_all()  -> int_ty_set {
    int_ty_set(INT_TY_SET_i8  | INT_TY_SET_u8 |
               INT_TY_SET_i16 | INT_TY_SET_u16 |
               INT_TY_SET_i32 | INT_TY_SET_u32 |
               INT_TY_SET_i64 | INT_TY_SET_u64 |
               INT_TY_SET_i   | INT_TY_SET_u)
}

fn intersection(a: int_ty_set, b: int_ty_set) -> int_ty_set {
    int_ty_set(*a & *b)
}

fn single_type_contained_in(tcx: ty::ctxt, a: int_ty_set) ->
    option<ty::t> {
    debug!{"single_type_contained_in(a=%s)", uint::to_str(*a, 10u)};

    if *a == INT_TY_SET_i8    { return some(ty::mk_i8(tcx)); }
    if *a == INT_TY_SET_u8    { return some(ty::mk_u8(tcx)); }
    if *a == INT_TY_SET_i16   { return some(ty::mk_i16(tcx)); }
    if *a == INT_TY_SET_u16   { return some(ty::mk_u16(tcx)); }
    if *a == INT_TY_SET_i32   { return some(ty::mk_i32(tcx)); }
    if *a == INT_TY_SET_u32   { return some(ty::mk_u32(tcx)); }
    if *a == INT_TY_SET_i64   { return some(ty::mk_i64(tcx)); }
    if *a == INT_TY_SET_u64   { return some(ty::mk_u64(tcx)); }
    if *a == INT_TY_SET_i     { return some(ty::mk_int(tcx)); }
    if *a == INT_TY_SET_u     { return some(ty::mk_uint(tcx)); }
    return none;
}

fn convert_integral_ty_to_int_ty_set(tcx: ty::ctxt, t: ty::t)
    -> int_ty_set {

    match get(t).struct {
      ty_int(int_ty) => match int_ty {
        ast::ty_i8   => int_ty_set(INT_TY_SET_i8),
        ast::ty_i16  => int_ty_set(INT_TY_SET_i16),
        ast::ty_i32  => int_ty_set(INT_TY_SET_i32),
        ast::ty_i64  => int_ty_set(INT_TY_SET_i64),
        ast::ty_i    => int_ty_set(INT_TY_SET_i),
        ast::ty_char => tcx.sess.bug(
            ~"char type passed to convert_integral_ty_to_int_ty_set()")
      },
      ty_uint(uint_ty) => match uint_ty {
        ast::ty_u8  => int_ty_set(INT_TY_SET_u8),
        ast::ty_u16 => int_ty_set(INT_TY_SET_u16),
        ast::ty_u32 => int_ty_set(INT_TY_SET_u32),
        ast::ty_u64 => int_ty_set(INT_TY_SET_u64),
        ast::ty_u   => int_ty_set(INT_TY_SET_u)
      },
      _ => tcx.sess.bug(~"non-integral type passed to \
                          convert_integral_ty_to_int_ty_set()")
    }
}

// Extra information needed to perform an assignment that may borrow.
// The `expr_id` and `span` are the id/span of the expression
// whose type is being assigned, and `borrow_scope` is the region
// scope to use if the value should be borrowed.
type assignment = {
    expr_id: ast::node_id,
    span: span,
    borrow_lb: ast::node_id,
};

type bound<T:copy> = option<T>;
type bounds<T:copy> = {lb: bound<T>, ub: bound<T>};

enum var_value<V:copy, T:copy> {
    redirect(V),
    root(T, uint),
}

struct vals_and_bindings<V:copy, T:copy> {
    vals: smallintmap<var_value<V, T>>;
    mut bindings: ~[(V, var_value<V, T>)];
}

struct node<V:copy, T:copy> {
    root: V;
    possible_types: T;
    rank: uint;
}

enum infer_ctxt = @{
    tcx: ty::ctxt,

    // We instantiate vals_and_bindings with bounds<ty::t> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    ty_var_bindings: vals_and_bindings<ty::tv_vid, bounds<ty::t>>,

    // The types that might instantiate an integral type variable are
    // represented by an int_ty_set.
    ty_var_integral_bindings: vals_and_bindings<ty::tvi_vid, int_ty_set>,

    // For region variables.
    region_var_bindings: vals_and_bindings<ty::region_vid,
                                           bounds<ty::region>>,

    // For keeping track of existing type and region variables.
    ty_var_counter: @mut uint,
    ty_var_integral_counter: @mut uint,
    region_var_counter: @mut uint,

    borrowings: DVec<{expr_id: ast::node_id,
                      span: span,
                      scope: ty::region,
                      mutbl: ast::mutability}>
};

enum fixup_err {
    unresolved_int_ty(tvi_vid),
    unresolved_ty(tv_vid),
    cyclic_ty(tv_vid),
    unresolved_region(region_vid),
    region_var_bound_by_region_var(region_vid, region_vid)
}

fn fixup_err_to_str(f: fixup_err) -> ~str {
    match f {
      unresolved_int_ty(_) => ~"unconstrained integral type",
      unresolved_ty(_) => ~"unconstrained type",
      cyclic_ty(_) => ~"cyclic type of infinite size",
      unresolved_region(_) => ~"unconstrained region",
      region_var_bound_by_region_var(r1, r2) => {
        fmt!{"region var %? bound by another region var %?; this is \
              a bug in rustc", r1, r2}
      }
    }
}

type ures = result::result<(), ty::type_err>;
type fres<T> = result::result<T, fixup_err>;

fn new_vals_and_bindings<V:copy, T:copy>() -> vals_and_bindings<V, T> {
    vals_and_bindings {
        vals: smallintmap::mk(),
        mut bindings: ~[]
    }
}

fn new_infer_ctxt(tcx: ty::ctxt) -> infer_ctxt {
    infer_ctxt(@{tcx: tcx,
                 ty_var_bindings: new_vals_and_bindings(),
                 ty_var_integral_bindings: new_vals_and_bindings(),
                 region_var_bindings: new_vals_and_bindings(),
                 ty_var_counter: @mut 0u,
                 ty_var_integral_counter: @mut 0u,
                 region_var_counter: @mut 0u,
                 borrowings: dvec()})}

fn mk_subty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    debug!{"mk_subty(%s <: %s)", a.to_str(cx), b.to_str(cx)};
    indent(|| cx.commit(|| (&sub(cx)).tys(a, b) ) ).to_ures()
}

fn can_mk_subty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    debug!{"can_mk_subty(%s <: %s)", a.to_str(cx), b.to_str(cx)};
    indent(|| cx.probe(|| (&sub(cx)).tys(a, b) ) ).to_ures()
}

fn mk_subr(cx: infer_ctxt, a: ty::region, b: ty::region) -> ures {
    debug!{"mk_subr(%s <: %s)", a.to_str(cx), b.to_str(cx)};
    indent(|| cx.commit(|| (&sub(cx)).regions(a, b) ) ).to_ures()
}

fn mk_eqty(cx: infer_ctxt, a: ty::t, b: ty::t) -> ures {
    debug!{"mk_eqty(%s <: %s)", a.to_str(cx), b.to_str(cx)};
    indent(|| cx.commit(|| cx.eq_tys(a, b) ) ).to_ures()
}

fn mk_assignty(cx: infer_ctxt, anmnt: &assignment,
               a: ty::t, b: ty::t) -> ures {
    debug!{"mk_assignty(%? / %s <: %s)",
           anmnt, a.to_str(cx), b.to_str(cx)};
    indent(|| cx.commit(||
        cx.assign_tys(anmnt, a, b)
    ) ).to_ures()
}

fn can_mk_assignty(cx: infer_ctxt, anmnt: &assignment,
                a: ty::t, b: ty::t) -> ures {
    debug!{"can_mk_assignty(%? / %s <: %s)",
           anmnt, a.to_str(cx), b.to_str(cx)};

    // FIXME(#2593)---this will not unroll any entries we make in the
    // borrowings table.  But this is OK for the moment because this
    // is only used in method lookup, and there must be exactly one
    // match or an error is reported. Still, it should be fixed. (#2593)
    // NDM OUTDATED

    indent(|| cx.probe(||
        cx.assign_tys(anmnt, a, b)
    ) ).to_ures()
}

// See comment on the type `resolve_state` below
fn resolve_type(cx: infer_ctxt, a: ty::t, modes: uint)
    -> fres<ty::t> {
    resolver(cx, modes).resolve_type_chk(a)
}

fn resolve_region(cx: infer_ctxt, r: ty::region, modes: uint)
    -> fres<ty::region> {
    resolver(cx, modes).resolve_region_chk(r)
}

fn resolve_borrowings(cx: infer_ctxt) {
    for cx.borrowings.each |item| {
        match resolve_region(cx, item.scope, resolve_all|force_all) {
          ok(region) => {
            debug!{"borrowing for expr %d resolved to region %?, mutbl %?",
                   item.expr_id, region, item.mutbl};
            cx.tcx.borrowings.insert(
                item.expr_id, {region: region, mutbl: item.mutbl});
          }

          err(e) => {
            let str = fixup_err_to_str(e);
            cx.tcx.sess.span_err(
                item.span,
                fmt!{"could not resolve lifetime for borrow: %s", str});
          }
        }
    }
}

trait then {
    fn then<T:copy>(f: fn() -> result<T,ty::type_err>)
        -> result<T,ty::type_err>;
}

impl ures: then {
    fn then<T:copy>(f: fn() -> result<T,ty::type_err>)
        -> result<T,ty::type_err> {
        self.chain(|_i| f())
    }
}

trait cres_helpers<T> {
    fn to_ures() -> ures;
    fn compare(t: T, f: fn() -> ty::type_err) -> cres<T>;
}

impl<T:copy> cres<T>: cres_helpers<T> {
    fn to_ures() -> ures {
        match self {
          ok(_v) => ok(()),
          err(e) => err(e)
        }
    }

    fn compare(t: T, f: fn() -> ty::type_err) -> cres<T> {
        do self.chain |s| {
            if s == t {
                self
            } else {
                err(f())
            }
        }
    }
}

trait to_str {
    fn to_str(cx: infer_ctxt) -> ~str;
}

impl ty::t: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        ty_to_str(cx.tcx, self)
    }
}

impl ty::mt: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        mt_to_str(cx.tcx, self)
    }
}

impl ty::region: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        util::ppaux::region_to_str(cx.tcx, self)
    }
}

impl<V:copy to_str> bound<V>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          some(v) => v.to_str(cx),
          none => ~"none"
        }
    }
}

impl<T:copy to_str> bounds<T>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        fmt!{"{%s <: %s}",
             self.lb.to_str(cx),
             self.ub.to_str(cx)}
    }
}

impl int_ty_set: to_str {
    fn to_str(_cx: infer_ctxt) -> ~str {
        match self {
          int_ty_set(v) => uint::to_str(v, 10u)
        }
    }
}

impl<V:copy vid, T:copy to_str> var_value<V, T>: to_str {
    fn to_str(cx: infer_ctxt) -> ~str {
        match self {
          redirect(vid) => fmt!{"redirect(%s)", vid.to_str()},
          root(pt, rk) => fmt!{"root(%s, %s)", pt.to_str(cx),
                               uint::to_str(rk, 10u)}
        }
    }
}

trait st {
    fn sub(infcx: infer_ctxt, b: self) -> ures;
    fn lub(infcx: infer_ctxt, b: self) -> cres<self>;
    fn glb(infcx: infer_ctxt, b: self) -> cres<self>;
}

impl ty::t: st {
    fn sub(infcx: infer_ctxt, &&b: ty::t) -> ures {
        (&sub(infcx)).tys(self, b).to_ures()
    }

    fn lub(infcx: infer_ctxt, &&b: ty::t) -> cres<ty::t> {
        (&lub(infcx)).tys(self, b)
    }

    fn glb(infcx: infer_ctxt, &&b: ty::t) -> cres<ty::t> {
        (&glb(infcx)).tys(self, b)
    }
}

impl ty::region: st {
    fn sub(infcx: infer_ctxt, &&b: ty::region) -> ures {
        (&sub(infcx)).regions(self, b).chain(|_r| ok(()))
    }

    fn lub(infcx: infer_ctxt, &&b: ty::region) -> cres<ty::region> {
        (&lub(infcx)).regions(self, b)
    }

    fn glb(infcx: infer_ctxt, &&b: ty::region) -> cres<ty::region> {
        (&glb(infcx)).regions(self, b)
    }
}

fn uok() -> ures {
    ok(())
}

fn rollback_to<V:copy vid, T:copy>(
    vb: &vals_and_bindings<V, T>, len: uint) {

    while vb.bindings.len() != len {
        let (vid, old_v) = vec::pop(vb.bindings);
        vb.vals.insert(vid.to_uint(), old_v);
    }
}

impl infer_ctxt {
    /// Execute `f` and commit the bindings if successful
    fn commit<T,E>(f: fn() -> result<T,E>) -> result<T,E> {

        assert self.ty_var_bindings.bindings.len() == 0u;
        assert self.region_var_bindings.bindings.len() == 0u;

        let r <- self.try(f);

        // FIXME (#2814)---could use a vec::clear() that ran destructors but
        // kept the vec at its currently allocated length
        self.ty_var_bindings.bindings = ~[];
        self.region_var_bindings.bindings = ~[];

        return r;
    }

    /// Execute `f`, unroll bindings on failure
    fn try<T,E>(f: fn() -> result<T,E>) -> result<T,E> {

        let tvbl = self.ty_var_bindings.bindings.len();
        let rbl = self.region_var_bindings.bindings.len();
        let bl = self.borrowings.len();

        debug!{"try(tvbl=%u, rbl=%u)", tvbl, rbl};
        let r <- f();
        match r {
          result::ok(_) => debug!{"try--ok"},
          result::err(_) => {
            debug!{"try--rollback"};
            rollback_to(&self.ty_var_bindings, tvbl);
            rollback_to(&self.region_var_bindings, rbl);
            while self.borrowings.len() != bl { self.borrowings.pop(); }
          }
        }
        return r;
    }

    /// Execute `f` then unroll any bindings it creates
    fn probe<T,E>(f: fn() -> result<T,E>) -> result<T,E> {
        assert self.ty_var_bindings.bindings.len() == 0u;
        assert self.region_var_bindings.bindings.len() == 0u;
        let r <- f();
        rollback_to(&self.ty_var_bindings, 0u);
        rollback_to(&self.region_var_bindings, 0u);
        return r;
    }
}

impl infer_ctxt {
    fn next_ty_var_id() -> tv_vid {
        let id = *self.ty_var_counter;
        *self.ty_var_counter += 1u;
        self.ty_var_bindings.vals.insert(id,
                             root({lb: none, ub: none}, 0u));
        return tv_vid(id);
    }

    fn next_ty_var() -> ty::t {
        ty::mk_var(self.tcx, self.next_ty_var_id())
    }

    fn next_ty_vars(n: uint) -> ~[ty::t] {
        vec::from_fn(n, |_i| self.next_ty_var())
    }

    fn next_ty_var_integral_id() -> tvi_vid {
        let id = *self.ty_var_integral_counter;
        *self.ty_var_integral_counter += 1u;

        self.ty_var_integral_bindings.vals.insert(id,
                              root(int_ty_set_all(), 0u));
        return tvi_vid(id);
    }

    fn next_ty_var_integral() -> ty::t {
        ty::mk_var_integral(self.tcx, self.next_ty_var_integral_id())
    }

    fn next_region_var_id(bnds: bounds<ty::region>) -> region_vid {
        let id = *self.region_var_counter;
        *self.region_var_counter += 1u;
        self.region_var_bindings.vals.insert(id, root(bnds, 0));
        return region_vid(id);
    }

    fn next_region_var_with_scope_lb(scope_id: ast::node_id) -> ty::region {
        self.next_region_var({lb: some(ty::re_scope(scope_id)),
                              ub: none})
    }

    fn next_region_var(bnds: bounds<ty::region>) -> ty::region {
        ty::re_var(self.next_region_var_id(bnds))
    }

    fn next_region_var_nb() -> ty::region { // nb == "no bounds"
        self.next_region_var({lb: none, ub: none})
    }

    fn ty_to_str(t: ty::t) -> ~str {
        ty_to_str(self.tcx,
                  self.resolve_type_vars_if_possible(t))
    }

    fn resolve_type_vars_if_possible(typ: ty::t) -> ty::t {
        match resolve_type(self, typ, resolve_all) {
          result::ok(new_type) => return new_type,
          result::err(_) => return typ
        }
    }

    fn resolve_region_if_possible(oldr: ty::region) -> ty::region {
        match resolve_region(self, oldr, resolve_all) {
          result::ok(newr) => return newr,
          result::err(_) => return oldr
        }
    }
}

impl infer_ctxt {

    fn set<V:copy vid, T:copy to_str>(
        vb: &vals_and_bindings<V, T>, vid: V,
        +new_v: var_value<V, T>) {

        let old_v = vb.vals.get(vid.to_uint());
        vec::push(vb.bindings, (vid, old_v));
        vb.vals.insert(vid.to_uint(), new_v);

        debug!{"Updating variable %s from %s to %s",
               vid.to_str(), old_v.to_str(self), new_v.to_str(self)};
    }

    fn get<V:copy vid, T:copy>(
        vb: &vals_and_bindings<V, T>, vid: V)
        -> node<V, T> {

        let vid_u = vid.to_uint();
        match vb.vals.find(vid_u) {
          none => {
            self.tcx.sess.bug(fmt!{"failed lookup of vid `%u`", vid_u});
          }
          some(var_val) => {
            match var_val {
              redirect(vid) => {
                let node = self.get(vb, vid);
                if node.root != vid {
                    // Path compression
                    vb.vals.insert(vid.to_uint(), redirect(node.root));
                }
                node
              }
              root(pt, rk) => {
                node {root: vid, possible_types: pt, rank: rk}
              }
            }
          }
        }
    }

    // Combines the two bounds into a more general bound.
    fn merge_bnd<V:copy to_str>(
        a: bound<V>, b: bound<V>,
        merge_op: fn(V,V) -> cres<V>) -> cres<bound<V>> {

        debug!{"merge_bnd(%s,%s)", a.to_str(self), b.to_str(self)};
        let _r = indenter();

        match (a, b) {
          (none, none) => ok(none),
          (some(_), none) => ok(a),
          (none, some(_)) => ok(b),
          (some(v_a), some(v_b)) => {
            do merge_op(v_a, v_b).chain |v| {
                ok(some(v))
            }
          }
        }
    }

    fn merge_bnds<V:copy to_str>(
        a: bounds<V>, b: bounds<V>,
        lub: fn(V,V) -> cres<V>,
        glb: fn(V,V) -> cres<V>) -> cres<bounds<V>> {

        let _r = indenter();
        do self.merge_bnd(a.ub, b.ub, glb).chain |ub| {
            debug!{"glb of ubs %s and %s is %s",
                   a.ub.to_str(self), b.ub.to_str(self),
                   ub.to_str(self)};
            do self.merge_bnd(a.lb, b.lb, lub).chain |lb| {
                debug!{"lub of lbs %s and %s is %s",
                       a.lb.to_str(self), b.lb.to_str(self),
                       lb.to_str(self)};
                ok({lb: lb, ub: ub})
            }
        }
    }

    // Updates the bounds for the variable `v_id` to be the intersection
    // of `a` and `b`.  That is, the new bounds for `v_id` will be
    // a bounds c such that:
    //    c.ub <: a.ub
    //    c.ub <: b.ub
    //    a.lb <: c.lb
    //    b.lb <: c.lb
    // If this cannot be achieved, the result is failure.

    fn set_var_to_merged_bounds<V:copy vid, T:copy to_str st>(
        vb: &vals_and_bindings<V, bounds<T>>,
        v_id: V, a: bounds<T>, b: bounds<T>, rank: uint) -> ures {

        // Think of the two diamonds, we want to find the
        // intersection.  There are basically four possibilities (you
        // can swap A/B in these pictures):
        //
        //       A         A
        //      / \       / \
        //     / B \     / B \
        //    / / \ \   / / \ \
        //   * *   * * * /   * *
        //    \ \ / /   \   / /
        //     \ B /   / \ / /
        //      \ /   *   \ /
        //       A     \ / A
        //              B

        debug!{"merge(%s,%s,%s)",
               v_id.to_str(),
               a.to_str(self),
               b.to_str(self)};

        // First, relate the lower/upper bounds of A and B.
        // Note that these relations *must* hold for us to
        // to be able to merge A and B at all, and relating
        // them explicitly gives the type inferencer more
        // information and helps to produce tighter bounds
        // when necessary.
        do indent {
        do self.bnds(a.lb, b.ub).then {
        do self.bnds(b.lb, a.ub).then {
        do self.merge_bnd(a.ub, b.ub, |x, y| x.glb(self, y) ).chain |ub| {
        do self.merge_bnd(a.lb, b.lb, |x, y| x.lub(self, y) ).chain |lb| {
            let bnds = {lb: lb, ub: ub};
            debug!{"merge(%s): bnds=%s",
                   v_id.to_str(),
                   bnds.to_str(self)};

            // the new bounds must themselves
            // be relatable:
            do self.bnds(bnds.lb, bnds.ub).then {
                self.set(vb, v_id, root(bnds, rank));
                uok()
            }
        }}}}}
    }

    /// Ensure that variable A is a subtype of variable B.  This is a
    /// subtle and tricky process, as described in detail at the top
    /// of this file.
    fn var_sub_var<V:copy vid, T:copy to_str st>(
        vb: &vals_and_bindings<V, bounds<T>>,
        a_id: V, b_id: V) -> ures {

        // Need to make sub_id a subtype of sup_id.
        let nde_a = self.get(vb, a_id);
        let nde_b = self.get(vb, b_id);
        let a_id = nde_a.root;
        let b_id = nde_b.root;
        let a_bounds = nde_a.possible_types;
        let b_bounds = nde_b.possible_types;

        debug!{"vars(%s=%s <: %s=%s)",
               a_id.to_str(), a_bounds.to_str(self),
               b_id.to_str(), b_bounds.to_str(self)};

        if a_id == b_id { return uok(); }

        // If both A's UB and B's LB have already been bound to types,
        // see if we can make those types subtypes.
        match (a_bounds.ub, b_bounds.lb) {
          (some(a_ub), some(b_lb)) => {
            let r = self.try(|| a_ub.sub(self, b_lb));
            match r {
              ok(()) => return result::ok(()),
              err(_) => { /*fallthrough */ }
            }
          }
          _ => { /*fallthrough*/ }
        }

        // Otherwise, we need to merge A and B so as to guarantee that
        // A remains a subtype of B.  Actually, there are other options,
        // but that's the route we choose to take.

        // Rank optimization

        // Make the node with greater rank the parent of the node with
        // smaller rank.
        if nde_a.rank > nde_b.rank {
            debug!{"vars(): a has smaller rank"};
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(vb, b_id, redirect(a_id));
            self.set_var_to_merged_bounds(
                vb, a_id, a_bounds, b_bounds, nde_a.rank).then(|| uok() )
        } else if nde_a.rank < nde_b.rank {
            debug!{"vars(): b has smaller rank"};
            // b has greater rank, so a should redirect to b.
            self.set(vb, a_id, redirect(b_id));
            self.set_var_to_merged_bounds(
                vb, b_id, a_bounds, b_bounds, nde_b.rank).then(|| uok() )
        } else {
            debug!{"vars(): a and b have equal rank"};
            assert nde_a.rank == nde_b.rank;
            // If equal, just redirect one to the other and increment
            // the other's rank.  We choose arbitrarily to redirect b
            // to a and increment a's rank.
            self.set(vb, b_id, redirect(a_id));
            self.set_var_to_merged_bounds(
                vb, a_id, a_bounds, b_bounds, nde_a.rank + 1u
            ).then(|| uok() )
        }
    }

    fn vars_integral<V:copy vid>(
        vb: &vals_and_bindings<V, int_ty_set>,
        a_id: V, b_id: V) -> ures {

        let nde_a = self.get(vb, a_id);
        let nde_b = self.get(vb, b_id);
        let a_id = nde_a.root;
        let b_id = nde_b.root;
        let a_pt = nde_a.possible_types;
        let b_pt = nde_b.possible_types;

        // If we're already dealing with the same two variables,
        // there's nothing to do.
        if a_id == b_id { return uok(); }

        // Otherwise, take the intersection of the two sets of
        // possible types.
        let intersection = intersection(a_pt, b_pt);
        if *intersection == INT_TY_SET_EMPTY {
            return err(ty::terr_no_integral_type);
        }

        // Rank optimization
        if nde_a.rank > nde_b.rank {
            debug!{"vars_integral(): a has smaller rank"};
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(vb, a_id, root(intersection, nde_a.rank));
            self.set(vb, b_id, redirect(a_id));
        } else if nde_a.rank < nde_b.rank {
            debug!{"vars_integral(): b has smaller rank"};
            // b has greater rank, so a should redirect to b.
            self.set(vb, b_id, root(intersection, nde_b.rank));
            self.set(vb, a_id, redirect(b_id));
        } else {
            debug!{"vars_integral(): a and b have equal rank"};
            assert nde_a.rank == nde_b.rank;
            // If equal, just redirect one to the other and increment
            // the other's rank.  We choose arbitrarily to redirect b
            // to a and increment a's rank.
            self.set(vb, a_id, root(intersection, nde_a.rank + 1u));
            self.set(vb, b_id, redirect(a_id));
        };

        uok()
    }

    /// make variable a subtype of T
    fn var_sub_t<V: copy vid, T: copy to_str st>(
        vb: &vals_and_bindings<V, bounds<T>>,
        a_id: V, b: T) -> ures {

        let nde_a = self.get(vb, a_id);
        let a_id = nde_a.root;
        let a_bounds = nde_a.possible_types;

        debug!{"var_sub_t(%s=%s <: %s)",
               a_id.to_str(), a_bounds.to_str(self),
               b.to_str(self)};
        let b_bounds = {lb: none, ub: some(b)};
        self.set_var_to_merged_bounds(vb, a_id, a_bounds, b_bounds,
                                      nde_a.rank)
    }

    fn var_integral_sub_t<V: copy vid>(
        vb: &vals_and_bindings<V, int_ty_set>,
        a_id: V, b: ty::t) -> ures {

        assert ty::type_is_integral(b);

        let nde_a = self.get(vb, a_id);
        let a_id = nde_a.root;
        let a_pt = nde_a.possible_types;

        let intersection =
            intersection(a_pt, convert_integral_ty_to_int_ty_set(
                self.tcx, b));
        if *intersection == INT_TY_SET_EMPTY {
            return err(ty::terr_no_integral_type);
        }
        self.set(vb, a_id, root(intersection, nde_a.rank));
        uok()
    }

    /// make T a subtype of variable
    fn t_sub_var<V: copy vid, T: copy to_str st>(
        vb: &vals_and_bindings<V, bounds<T>>,
        a: T, b_id: V) -> ures {

        let a_bounds = {lb: some(a), ub: none};
        let nde_b = self.get(vb, b_id);
        let b_id = nde_b.root;
        let b_bounds = nde_b.possible_types;

        debug!{"t_sub_var(%s <: %s=%s)",
               a.to_str(self),
               b_id.to_str(), b_bounds.to_str(self)};
        self.set_var_to_merged_bounds(vb, b_id, a_bounds, b_bounds,
                                      nde_b.rank)
    }

    fn t_sub_var_integral<V: copy vid>(
        vb: &vals_and_bindings<V, int_ty_set>,
        a: ty::t, b_id: V) -> ures {

        assert ty::type_is_integral(a);

        let nde_b = self.get(vb, b_id);
        let b_id = nde_b.root;
        let b_pt = nde_b.possible_types;

        let intersection =
            intersection(b_pt, convert_integral_ty_to_int_ty_set(
                self.tcx, a));
        if *intersection == INT_TY_SET_EMPTY {
            return err(ty::terr_no_integral_type);
        }
        self.set(vb, b_id, root(intersection, nde_b.rank));
        uok()
    }

    fn bnds<T:copy to_str st>(
        a: bound<T>, b: bound<T>) -> ures {

        debug!{"bnds(%s <: %s)", a.to_str(self), b.to_str(self)};
        do indent {
            match (a, b) {
              (none, none) |
              (some(_), none) |
              (none, some(_)) => {
                uok()
              }
              (some(t_a), some(t_b)) => {
                t_a.sub(self, t_b)
              }
            }
        }
    }

    fn sub_tys(a: ty::t, b: ty::t) -> ures {
        (&sub(self)).tys(a, b).chain(|_t| ok(()) )
    }

    fn sub_regions(a: ty::region, b: ty::region) -> ures {
        (&sub(self)).regions(a, b).chain(|_t| ok(()) )
    }

    fn eq_tys(a: ty::t, b: ty::t) -> ures {
        self.sub_tys(a, b).then(|| {
            self.sub_tys(b, a)
        })
    }

    fn eq_regions(a: ty::region, b: ty::region) -> ures {
        debug!{"eq_regions(%s, %s)",
               a.to_str(self), b.to_str(self)};
        do indent {
            self.try(|| {
                do self.sub_regions(a, b).then {
                    self.sub_regions(b, a)
                }
            }).chain_err(|e| {
                // substitute a better error, but use the regions
                // found in the original error
                match e {
                  ty::terr_regions_does_not_outlive(a1, b1) =>
                    err(ty::terr_regions_not_same(a1, b1)),
                  _ => err(e)
                }
            })
        }
    }
}

// Resolution is the process of removing type variables and replacing
// them with their inferred values.  Unfortunately our inference has
// become fairly complex and so there are a number of options to
// control *just how much* you want to resolve and how you want to do
// it.
//
// # Controlling the scope of resolution
//
// The options resolve_* determine what kinds of variables get
// resolved.  Generally resolution starts with a top-level type
// variable; we will always resolve this.  However, once we have
// resolved that variable, we may end up with a type that still
// contains type variables.  For example, if we resolve `<T0>` we may
// end up with something like `[<T1>]`.  If the option
// `resolve_nested_tvar` is passed, we will then go and recursively
// resolve `<T1>`.
//
// The options `resolve_rvar` and `resolve_ivar` control whether we
// resolve region and integral variables, respectively.
//
// # What do if things are unconstrained
//
// Sometimes we will encounter a variable that has no constraints, and
// therefore cannot sensibly be mapped to any particular result.  By
// default, we will leave such variables as is (so you will get back a
// variable in your result).  The options force_* will cause the
// resolution to fail in this case intead, except for the case of
// integral variables, which resolve to `int` if forced.
//
// # resolve_all and force_all
//
// The options are a bit set, so you can use the *_all to resolve or
// force all kinds of variables (including those we may add in the
// future).  If you want to resolve everything but one type, you are
// probably better off writing `resolve_all - resolve_ivar`.

const resolve_nested_tvar: uint = 0b00000001;
const resolve_rvar: uint        = 0b00000010;
const resolve_ivar: uint        = 0b00000100;
const resolve_all: uint         = 0b00000111;
const force_tvar: uint          = 0b00010000;
const force_rvar: uint          = 0b00100000;
const force_ivar: uint          = 0b01000000;
const force_all: uint           = 0b01110000;

type resolve_state_ = {
    infcx: infer_ctxt,
    modes: uint,
    mut err: option<fixup_err>,
    mut v_seen: ~[tv_vid]
};

enum resolve_state {
    resolve_state_(@resolve_state_)
}

fn resolver(infcx: infer_ctxt, modes: uint) -> resolve_state {
    resolve_state_(@{infcx: infcx,
                     modes: modes,
                     mut err: none,
                     mut v_seen: ~[]})
}

impl resolve_state {
    fn should(mode: uint) -> bool {
        (self.modes & mode) == mode
    }

    fn resolve_type_chk(typ: ty::t) -> fres<ty::t> {
        self.err = none;

        debug!{"Resolving %s (modes=%x)",
               ty_to_str(self.infcx.tcx, typ),
               self.modes};

        // n.b. This is a hokey mess because the current fold doesn't
        // allow us to pass back errors in any useful way.

        assert vec::is_empty(self.v_seen);
        let rty = indent(|| self.resolve_type(typ) );
        assert vec::is_empty(self.v_seen);
        match self.err {
          none => {
            debug!{"Resolved to %s (modes=%x)",
                   ty_to_str(self.infcx.tcx, rty),
                   self.modes};
            return ok(rty);
          }
          some(e) => return err(e)
        }
    }

    fn resolve_region_chk(orig: ty::region) -> fres<ty::region> {
        self.err = none;
        let resolved = indent(|| self.resolve_region(orig) );
        match self.err {
          none => ok(resolved),
          some(e) => err(e)
        }
    }

    fn resolve_type(typ: ty::t) -> ty::t {
        debug!{"resolve_type(%s)", typ.to_str(self.infcx)};
        indent(fn&() -> ty::t {
            if !ty::type_needs_infer(typ) { return typ; }

            match ty::get(typ).struct {
              ty::ty_var(vid) => {
                self.resolve_ty_var(vid)
              }
              ty::ty_var_integral(vid) => {
                self.resolve_ty_var_integral(vid)
              }
              _ => {
                if !self.should(resolve_rvar) &&
                    !self.should(resolve_nested_tvar) {
                    // shortcircuit for efficiency
                    typ
                } else {
                    ty::fold_regions_and_ty(
                        self.infcx.tcx, typ,
                        |r| self.resolve_region(r),
                        |t| self.resolve_nested_tvar(t),
                        |t| self.resolve_nested_tvar(t))
                }
              }
            }
        })
    }

    fn resolve_nested_tvar(typ: ty::t) -> ty::t {
        debug!{"Resolve_if_deep(%s)", typ.to_str(self.infcx)};
        if !self.should(resolve_nested_tvar) {
            typ
        } else {
            self.resolve_type(typ)
        }
    }

    fn resolve_region(orig: ty::region) -> ty::region {
        debug!{"Resolve_region(%s)", orig.to_str(self.infcx)};
        match orig {
          ty::re_var(rid) => self.resolve_region_var(rid),
          _ => orig
        }
    }

    fn resolve_region_var(rid: region_vid) -> ty::region {
        if !self.should(resolve_rvar) {
            return ty::re_var(rid)
        }
        let nde = self.infcx.get(&self.infcx.region_var_bindings, rid);
        let bounds = nde.possible_types;
        match bounds {
          { ub:_, lb:some(r) } => { self.assert_not_rvar(rid, r); r }
          { ub:some(r), lb:_ } => { self.assert_not_rvar(rid, r); r }
          { ub:none, lb:none } => {
            if self.should(force_rvar) {
                self.err = some(unresolved_region(rid));
            }
            ty::re_var(rid)
          }
        }
    }

    fn assert_not_rvar(rid: region_vid, r: ty::region) {
        match r {
          ty::re_var(rid2) => {
            self.err = some(region_var_bound_by_region_var(rid, rid2));
          }
          _ => { }
        }
    }

    fn resolve_ty_var(vid: tv_vid) -> ty::t {
        if vec::contains(self.v_seen, vid) {
            self.err = some(cyclic_ty(vid));
            return ty::mk_var(self.infcx.tcx, vid);
        } else {
            vec::push(self.v_seen, vid);
            let tcx = self.infcx.tcx;

            // Nonobvious: prefer the most specific type
            // (i.e., the lower bound) to the more general
            // one.  More general types in Rust (e.g., fn())
            // tend to carry more restrictions or higher
            // perf. penalties, so it pays to know more.

            let nde = self.infcx.get(&self.infcx.ty_var_bindings, vid);
            let bounds = nde.possible_types;

            let t1 = match bounds {
              { ub:_, lb:some(t) } if !type_is_bot(t) => self.resolve_type(t),
              { ub:some(t), lb:_ } => self.resolve_type(t),
              { ub:_, lb:some(t) } => self.resolve_type(t),
              { ub:none, lb:none } => {
                if self.should(force_tvar) {
                    self.err = some(unresolved_ty(vid));
                }
                ty::mk_var(tcx, vid)
              }
            };
            vec::pop(self.v_seen);
            return t1;
        }
    }

    fn resolve_ty_var_integral(vid: tvi_vid) -> ty::t {
        if !self.should(resolve_ivar) {
            return ty::mk_var_integral(self.infcx.tcx, vid);
        }

        let nde = self.infcx.get(&self.infcx.ty_var_integral_bindings, vid);
        let pt = nde.possible_types;

        // If there's only one type in the set of possible types, then
        // that's the answer.
        match single_type_contained_in(self.infcx.tcx, pt) {
          some(t) => t,
          none => {
            if self.should(force_ivar) {
                // As a last resort, default to int.
                let ty = ty::mk_int(self.infcx.tcx);
                self.infcx.set(
                    &self.infcx.ty_var_integral_bindings, vid,
                    root(convert_integral_ty_to_int_ty_set(self.infcx.tcx,
                                                           ty),
                        nde.rank));
                ty
            } else {
                ty::mk_var_integral(self.infcx.tcx, vid)
            }
          }
        }
    }
}

// ______________________________________________________________________
// Type assignment
//
// True if rvalues of type `a` can be assigned to lvalues of type `b`.
// This may cause borrowing to the region scope enclosing `a_node_id`.
//
// The strategy here is somewhat non-obvious.  The problem is
// that the constraint we wish to contend with is not a subtyping
// constraint.  Currently, for variables, we only track what it
// must be a subtype of, not what types it must be assignable to
// (or from).  Possibly, we should track that, but I leave that
// refactoring for another day.
//
// Instead, we look at each variable involved and try to extract
// *some* sort of bound.  Typically, the type a is the argument
// supplied to a call; it typically has a *lower bound* (which
// comes from having been assigned a value).  What we'd actually
// *like* here is an upper-bound, but we generally don't have
// one.  The type b is the expected type and it typically has a
// lower-bound too, which is good.
//
// The way we deal with the fact that we often don't have the
// bounds we need is to be a bit careful.  We try to get *some*
// bound from each side, preferring the upper from a and the
// lower from b.  If we fail to get a bound from both sides, then
// we just fall back to requiring that a <: b.
//
// Assuming we have a bound from both sides, we will then examine
// these bounds and see if they have the form (@M_a T_a, &rb.M_b T_b)
// (resp. ~M_a T_a, ~[M_a T_a], etc).  If they do not, we fall back to
// subtyping.
//
// If they *do*, then we know that the two types could never be
// subtypes of one another.  We will then construct a type @const T_b
// and ensure that type a is a subtype of that.  This allows for the
// possibility of assigning from a type like (say) @~[mut T1] to a type
// &~[T2] where T1 <: T2.  This might seem surprising, since the `@`
// points at mutable memory but the `&` points at immutable memory.
// This would in fact be unsound, except for the borrowck, which comes
// later and guarantees that such mutability conversions are safe.
// See borrowck for more details.  Next we require that the region for
// the enclosing scope be a superregion of the region r.
//
// You might wonder why we don't make the type &e.const T_a where e is
// the enclosing region and check that &e.const T_a <: B.  The reason
// is that the type of A is (generally) just a *lower-bound*, so this
// would be imposing that lower-bound also as the upper-bound on type
// A.  But this upper-bound might be stricter than what is truly
// needed.

impl infer_ctxt {
    fn assign_tys(anmnt: &assignment, a: ty::t, b: ty::t) -> ures {

        fn select(fst: option<ty::t>, snd: option<ty::t>) -> option<ty::t> {
            match fst {
              some(t) => some(t),
              none => match snd {
                some(t) => some(t),
                none => none
              }
            }
        }

        debug!{"assign_tys(anmnt=%?, %s -> %s)",
               anmnt, a.to_str(self), b.to_str(self)};
        let _r = indenter();

        match (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_bot, _) => {
            uok()
          }

          (ty::ty_var(a_id), ty::ty_var(b_id)) => {
            let nde_a = self.get(&self.ty_var_bindings, a_id);
            let nde_b = self.get(&self.ty_var_bindings, b_id);
            let a_bounds = nde_a.possible_types;
            let b_bounds = nde_b.possible_types;

            let a_bnd = select(a_bounds.ub, a_bounds.lb);
            let b_bnd = select(b_bounds.lb, b_bounds.ub);
            self.assign_tys_or_sub(anmnt, a, b, a_bnd, b_bnd)
          }

          (ty::ty_var(a_id), _) => {
            let nde_a = self.get(&self.ty_var_bindings, a_id);
            let a_bounds = nde_a.possible_types;

            let a_bnd = select(a_bounds.ub, a_bounds.lb);
            self.assign_tys_or_sub(anmnt, a, b, a_bnd, some(b))
          }

          (_, ty::ty_var(b_id)) => {
            let nde_b = self.get(&self.ty_var_bindings, b_id);
            let b_bounds = nde_b.possible_types;

            let b_bnd = select(b_bounds.lb, b_bounds.ub);
            self.assign_tys_or_sub(anmnt, a, b, some(a), b_bnd)
          }

          (_, _) => {
            self.assign_tys_or_sub(anmnt, a, b, some(a), some(b))
          }
        }
    }

    fn assign_tys_or_sub(
        anmnt: &assignment,
        a: ty::t, b: ty::t,
        +a_bnd: option<ty::t>, +b_bnd: option<ty::t>) -> ures {

        debug!{"assign_tys_or_sub(anmnt=%?, %s -> %s, %s -> %s)",
               anmnt, a.to_str(self), b.to_str(self),
               a_bnd.to_str(self), b_bnd.to_str(self)};
        let _r = indenter();

        fn is_borrowable(v: ty::vstore) -> bool {
            match v {
              ty::vstore_fixed(_) | ty::vstore_uniq | ty::vstore_box => true,
              ty::vstore_slice(_) => false
            }
        }

        match (a_bnd, b_bnd) {
          (some(a_bnd), some(b_bnd)) => {
            match (ty::get(a_bnd).struct, ty::get(b_bnd).struct) {
              (ty::ty_box(mt_a), ty::ty_rptr(r_b, mt_b)) => {
                let nr_b = ty::mk_box(self.tcx, {ty: mt_b.ty,
                                                 mutbl: m_const});
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }
              (ty::ty_uniq(mt_a), ty::ty_rptr(r_b, mt_b)) => {
                let nr_b = ty::mk_uniq(self.tcx, {ty: mt_b.ty,
                                                  mutbl: m_const});
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }
              (ty::ty_estr(vs_a),
               ty::ty_estr(ty::vstore_slice(r_b)))
              if is_borrowable(vs_a) => {
                let nr_b = ty::mk_estr(self.tcx, vs_a);
                self.crosspollinate(anmnt, a, nr_b, m_imm, r_b)
              }

              (ty::ty_evec(mt_a, vs_a),
               ty::ty_evec(mt_b, ty::vstore_slice(r_b)))
              if is_borrowable(vs_a) => {
                let nr_b = ty::mk_evec(self.tcx, {ty: mt_b.ty,
                                                  mutbl: m_const}, vs_a);
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }
              _ => {
                self.sub_tys(a, b)
              }
            }
          }
          _ => {
            self.sub_tys(a, b)
          }
        }
    }

    fn crosspollinate(anmnt: &assignment,
                      a: ty::t,
                      nr_b: ty::t,
                      m: ast::mutability,
                      r_b: ty::region) -> ures {

        debug!{"crosspollinate(anmnt=%?, a=%s, nr_b=%s, r_b=%s)",
               anmnt, a.to_str(self), nr_b.to_str(self),
               r_b.to_str(self)};

        do indent {
            do self.sub_tys(a, nr_b).then {
                // Create a fresh region variable `r_a` with the given
                // borrow bounds:
                let r_a = self.next_region_var_with_scope_lb(anmnt.borrow_lb);

                debug!{"anmnt=%?", anmnt};
                do (&sub(self)).contraregions(r_a, r_b).chain |_r| {
                    // if successful, add an entry indicating that
                    // borrowing occurred
                    debug!{"borrowing expression #%?, scope=%?, m=%?",
                           anmnt, r_a, m};
                    self.borrowings.push({expr_id: anmnt.expr_id,
                                          span: anmnt.span,
                                          scope: r_a,
                                          mutbl: m});
                    uok()
                }
            }
        }
    }
}

// ______________________________________________________________________
// Type combining
//
// There are three type combiners: sub, lub, and glb.  Each implements
// the trait `combine` and contains methods for combining two
// instances of various things and yielding a new instance.  These
// combiner methods always yield a `result<T>`---failure is propagated
// upward using `chain()` methods.
//
// There is a lot of common code for these operations, which is
// abstracted out into functions named `super_X()` which take a combiner
// instance as the first parameter.  This would be better implemented
// using traits.  For this system to work properly, you should not
// call the `super_X(foo, ...)` functions directly, but rather call
// `foo.X(...)`.  The implementation of `X()` can then choose to delegate
// to the `super` routine or to do other things.
//
// In reality, the sub operation is rather different from lub/glb, but
// they are combined into one trait to avoid duplication (they used to
// be separate but there were many bugs because there were two copies
// of most routines).
//
// The differences are:
//
// - when making two things have a sub relationship, the order of the
//   arguments is significant (a <: b) and the return value of the
//   combine functions is largely irrelevant.  The important thing is
//   whether the action succeeds or fails.  If it succeeds, then side
//   effects have been committed into the type variables.
//
// - for GLB/LUB, the order of arguments is not significant (GLB(a,b) ==
//   GLB(b,a)) and the return value is important (it is the GLB).  Of
//   course GLB/LUB may also have side effects.
//
// Contravariance
//
// When you are relating two things which have a contravariant
// relationship, you should use `contratys()` or `contraregions()`,
// rather than inversing the order of arguments!  This is necessary
// because the order of arguments is not relevant for LUB and GLB.  It
// is also useful to track which value is the "expected" value in
// terms of error reporting, although we do not do that properly right
// now.

type cres<T> = result<T,ty::type_err>;

trait combine {
    fn infcx() -> infer_ctxt;
    fn tag() -> ~str;

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt>;
    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tys(a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]>;
    fn self_tys(a: option<ty::t>, b: option<ty::t>) -> cres<option<ty::t>>;
    fn substs(as: &ty::substs, bs: &ty::substs) -> cres<ty::substs>;
    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty>;
    fn flds(a: ty::field, b: ty::field) -> cres<ty::field>;
    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode>;
    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg>;
    fn protos(p1: ty::fn_proto, p2: ty::fn_proto) -> cres<ty::fn_proto>;
    fn ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style>;
    fn purities(f1: purity, f2: purity) -> cres<purity>;
    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region>;
    fn regions(a: ty::region, b: ty::region) -> cres<ty::region>;
    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore>;
}

enum sub = infer_ctxt;  // "subtype", "subregion" etc
enum lub = infer_ctxt;  // "least upper bound" (common supertype)
enum glb = infer_ctxt;  // "greatest lower bound" (common subtype)

fn super_substs<C:combine>(
    self: &C, a: &ty::substs, b: &ty::substs) -> cres<ty::substs> {

    fn eq_opt_regions(infcx: infer_ctxt,
                      a: option<ty::region>,
                      b: option<ty::region>) -> cres<option<ty::region>> {
        match (a, b) {
          (none, none) => {
            ok(none)
          }
          (some(a), some(b)) => {
            do infcx.eq_regions(a, b).then {
                ok(some(a))
            }
          }
          (_, _) => {
            // If these two substitutions are for the same type (and
            // they should be), then the type should either
            // consistently have a region parameter or not have a
            // region parameter.
            infcx.tcx.sess.bug(
                fmt!{"substitution a had opt_region %s and \
                      b had opt_region %s",
                     a.to_str(infcx),
                     b.to_str(infcx)});
          }
        }
    }

    do self.tps(a.tps, b.tps).chain |tps| {
        do self.self_tys(a.self_ty, b.self_ty).chain |self_ty| {
            do eq_opt_regions(self.infcx(), a.self_r, b.self_r).chain
                |self_r| {
                ok({self_r: self_r, self_ty: self_ty, tps: tps})
            }
        }
    }
}

fn super_tps<C:combine>(
    self: &C, as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {

    // Note: type parameters are always treated as *invariant*
    // (otherwise the type system would be unsound).  In the
    // future we could allow type parameters to declare a
    // variance.

    if vec::same_length(as, bs) {
        iter_vec2(as, bs, |a, b| {
            self.infcx().eq_tys(a, b)
        }).then(|| ok(as.to_vec()) )
    } else {
        err(ty::terr_ty_param_size(bs.len(), as.len()))
    }
}

fn super_self_tys<C:combine>(
    self: &C, a: option<ty::t>, b: option<ty::t>) -> cres<option<ty::t>> {

    // Note: the self type parameter is (currently) always treated as
    // *invariant* (otherwise the type system would be unsound).

    match (a, b) {
      (none, none) => {
        ok(none)
      }
      (some(a), some(b)) => {
        self.infcx().eq_tys(a, b).then(|| ok(some(a)) )
      }
      (none, some(_)) |
      (some(_), none) => {
        // I think it should never happen that we unify two substs and
        // one of them has a self_ty and one doesn't...? I could be
        // wrong about this.
        err(ty::terr_self_substs)
      }
    }
}

fn super_flds<C:combine>(
    self: &C, a: ty::field, b: ty::field) -> cres<ty::field> {

    if a.ident == b.ident {
        self.mts(a.mt, b.mt)
            .chain(|mt| ok({ident: a.ident, mt: mt}) )
            .chain_err(|e| err(ty::terr_in_field(@e, a.ident)) )
    } else {
        err(ty::terr_record_fields(b.ident, a.ident))
    }
}

fn super_modes<C:combine>(
    self: &C, a: ast::mode, b: ast::mode)
    -> cres<ast::mode> {

    let tcx = self.infcx().tcx;
    ty::unify_mode(tcx, a, b)
}

fn super_args<C:combine>(
    self: &C, a: ty::arg, b: ty::arg)
    -> cres<ty::arg> {

    do self.modes(a.mode, b.mode).chain |m| {
        do self.contratys(a.ty, b.ty).chain |t| {
            ok({mode: m, ty: t})
        }
    }
}

fn super_vstores<C:combine>(
    self: &C, vk: ty::terr_vstore_kind,
    a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {

    match (a, b) {
      (ty::vstore_slice(a_r), ty::vstore_slice(b_r)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            ok(ty::vstore_slice(r))
        }
      }

      _ if a == b => {
        ok(a)
      }

      _ => {
        err(ty::terr_vstores_differ(vk, b, a))
      }
    }
}

fn super_fns<C:combine>(
    self: &C, a_f: &ty::fn_ty, b_f: &ty::fn_ty) -> cres<ty::fn_ty> {

    fn argvecs<C:combine>(self: &C, a_args: ~[ty::arg],
                          b_args: ~[ty::arg]) -> cres<~[ty::arg]> {

        if vec::same_length(a_args, b_args) {
            map_vec2(a_args, b_args, |a, b| self.args(a, b) )
        } else {
            err(ty::terr_arg_count)
        }
    }

    do self.protos(a_f.proto, b_f.proto).chain |p| {
        do self.ret_styles(a_f.ret_style, b_f.ret_style).chain |rs| {
            do argvecs(self, a_f.inputs, b_f.inputs).chain |inputs| {
                do self.tys(a_f.output, b_f.output).chain |output| {
                    do self.purities(a_f.purity, b_f.purity).chain |purity| {
                    // FIXME: uncomment if #2588 doesn't get accepted:
                    // self.infcx().constrvecs(a_f.constraints,
                    //                         b_f.constraints).then {||
                        ok({purity: purity,
                            proto: p,
                            bounds: a_f.bounds, // XXX: This is wrong!
                            inputs: inputs,
                            output: output,
                            ret_style: rs})
                    // }
                    }
                }
            }
        }
    }
}

fn super_tys<C:combine>(
    self: &C, a: ty::t, b: ty::t) -> cres<ty::t> {

    let tcx = self.infcx().tcx;
    match (ty::get(a).struct, ty::get(b).struct) {
      // The "subtype" ought to be handling cases involving bot or var:
      (ty::ty_bot, _) |
      (_, ty::ty_bot) |
      (ty::ty_var(_), _) |
      (_, ty::ty_var(_)) => {
        tcx.sess.bug(
            fmt!{"%s: bot and var types should have been handled (%s,%s)",
                 self.tag(),
                 a.to_str(self.infcx()),
                 b.to_str(self.infcx())});
      }

      // Have to handle these first
      (ty::ty_var_integral(a_id), ty::ty_var_integral(b_id)) => {
        self.infcx().vars_integral(&self.infcx().ty_var_integral_bindings,
                                   a_id, b_id)
            .then(|| ok(a) )
      }
      (ty::ty_var_integral(a_id), ty::ty_int(_)) |
      (ty::ty_var_integral(a_id), ty::ty_uint(_)) => {
        self.infcx().var_integral_sub_t(
            &self.infcx().ty_var_integral_bindings,
            a_id, b).then(|| ok(a) )
      }
      (ty::ty_int(_), ty::ty_var_integral(b_id)) |
      (ty::ty_uint(_), ty::ty_var_integral(b_id)) => {
        self.infcx().t_sub_var_integral(
            &self.infcx().ty_var_integral_bindings,
            a, b_id).then(|| ok(a) )
      }

      (ty::ty_int(_), _) |
      (ty::ty_uint(_), _) |
      (ty::ty_float(_), _) => {
        let as = ty::get(a).struct;
        let bs = ty::get(b).struct;
        if as == bs {
            ok(a)
        } else {
            err(ty::terr_sorts(b, a))
        }
      }

      (ty::ty_nil, _) |
      (ty::ty_bool, _) => {
        let cfg = tcx.sess.targ_cfg;
        if ty::mach_sty(cfg, a) == ty::mach_sty(cfg, b) {
            ok(a)
        } else {
            err(ty::terr_sorts(b, a))
        }
      }

      (ty::ty_param(a_p), ty::ty_param(b_p)) if a_p.idx == b_p.idx => {
        ok(a)
      }

      (ty::ty_enum(a_id, ref a_substs), ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_substs, b_substs).chain |tps| {
            ok(ty::mk_enum(tcx, a_id, tps))
        }
      }

      (ty::ty_trait(a_id, ref a_substs, a_vstore),
       ty::ty_trait(b_id, ref b_substs, b_vstore))
      if a_id == b_id => {
        do self.substs(a_substs, b_substs).chain |substs| {
            do self.vstores(ty::terr_trait, a_vstore,
                            b_vstore).chain |vstores| {
                ok(ty::mk_trait(tcx, a_id, substs, vstores))
            }
        }
      }

      (ty::ty_class(a_id, ref a_substs), ty::ty_class(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_substs, b_substs).chain |substs| {
            ok(ty::mk_class(tcx, a_id, substs))
        }
      }

      (ty::ty_box(a_mt), ty::ty_box(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_box(tcx, mt))
        }
      }

      (ty::ty_uniq(a_mt), ty::ty_uniq(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_uniq(tcx, mt))
        }
      }

      (ty::ty_ptr(a_mt), ty::ty_ptr(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_ptr(tcx, mt))
        }
      }

      (ty::ty_rptr(a_r, a_mt), ty::ty_rptr(b_r, b_mt)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            do self.mts(a_mt, b_mt).chain |mt| {
                ok(ty::mk_rptr(tcx, r, mt))
            }
        }
      }

      (ty::ty_evec(a_mt, vs_a), ty::ty_evec(b_mt, vs_b)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            do self.vstores(ty::terr_vec, vs_a, vs_b).chain |vs| {
                ok(ty::mk_evec(tcx, mt, vs))
            }
        }
      }

      (ty::ty_estr(vs_a), ty::ty_estr(vs_b)) => {
        do self.vstores(ty::terr_str, vs_a, vs_b).chain |vs| {
            ok(ty::mk_estr(tcx,vs))
        }
      }

      (ty::ty_rec(as), ty::ty_rec(bs)) => {
        if vec::same_length(as, bs) {
            map_vec2(as, bs, |a,b| {
                self.flds(a, b)
            }).chain(|flds| ok(ty::mk_rec(tcx, flds)) )
        } else {
            err(ty::terr_record_size(bs.len(), as.len()))
        }
      }

      (ty::ty_tup(as), ty::ty_tup(bs)) => {
        if vec::same_length(as, bs) {
            map_vec2(as, bs, |a, b| self.tys(a, b) )
                .chain(|ts| ok(ty::mk_tup(tcx, ts)) )
        } else {
            err(ty::terr_tuple_size(bs.len(), as.len()))
        }
      }

      (ty::ty_fn(ref a_fty), ty::ty_fn(ref b_fty)) => {
        do self.fns(a_fty, b_fty).chain |fty| {
            ok(ty::mk_fn(tcx, fty))
        }
      }

      _ => err(ty::terr_sorts(b, a))
    }
}

impl sub: combine {
    fn infcx() -> infer_ctxt { *self }
    fn tag() -> ~str { ~"sub" }

    fn lub() -> lub { lub(self.infcx()) }

    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        self.tys(b, a)
    }

    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region> {
        self.regions(b, a)
    }

    fn regions(a: ty::region, b: ty::region) -> cres<ty::region> {
        debug!{"%s.regions(%s, %s)",
               self.tag(),
               a.to_str(self.infcx()),
               b.to_str(self.infcx())};
        do indent {
            match (a, b) {
              (ty::re_var(a_id), ty::re_var(b_id)) => {
                do self.infcx().var_sub_var(&self.region_var_bindings,
                                            a_id, b_id).then {
                    ok(a)
                }
              }
              (ty::re_var(a_id), _) => {
                do self.infcx().var_sub_t(&self.region_var_bindings,
                                          a_id, b).then {
                      ok(a)
                  }
              }
              (_, ty::re_var(b_id)) => {
                  do self.infcx().t_sub_var(&self.region_var_bindings,
                                            a, b_id).then {
                      ok(a)
                  }
              }
              _ => {
                  do (&self.lub()).regions(a, b).compare(b) {
                    ty::terr_regions_does_not_outlive(b, a)
                  }
              }
            }
        }
    }

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        debug!{"mts(%s <: %s)", a.to_str(*self), b.to_str(*self)};

        if a.mutbl != b.mutbl && b.mutbl != m_const {
            return err(ty::terr_mutability);
        }

        match b.mutbl {
          m_mutbl => {
            // If supertype is mut, subtype must match exactly
            // (i.e., invariant if mut):
            self.infcx().eq_tys(a.ty, b.ty).then(|| ok(a) )
          }
          m_imm | m_const => {
            // Otherwise we can be covariant:
            self.tys(a.ty, b.ty).chain(|_t| ok(a) )
          }
        }
    }

    fn protos(a: ty::fn_proto, b: ty::fn_proto) -> cres<ty::fn_proto> {
        match (a, b) {
            (ty::proto_bare, _) => ok(ty::proto_bare),

            (ty::proto_vstore(ty::vstore_box),
             ty::proto_vstore(ty::vstore_slice(_))) =>
                ok(ty::proto_vstore(ty::vstore_box)),

            (ty::proto_vstore(ty::vstore_uniq),
             ty::proto_vstore(ty::vstore_slice(_))) =>
                ok(ty::proto_vstore(ty::vstore_uniq)),

            (_, ty::proto_bare) => err(ty::terr_proto_mismatch(b, a)),
            (ty::proto_vstore(vs_a), ty::proto_vstore(vs_b)) => {
                do self.vstores(ty::terr_fn, vs_a, vs_b).chain |vs_c| {
                    ok(ty::proto_vstore(vs_c))
                }
            }
        }
    }

    fn purities(f1: purity, f2: purity) -> cres<purity> {
        (&self.lub()).purities(f1, f2).compare(f2, || {
            ty::terr_purity_mismatch(f2, f1)
        })
    }

    fn ret_styles(a: ret_style, b: ret_style) -> cres<ret_style> {
        (&self.lub()).ret_styles(a, b).compare(b, || {
            ty::terr_ret_style_mismatch(b, a)
        })
    }

    fn tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        debug!{"%s.tys(%s, %s)", self.tag(),
               a.to_str(*self), b.to_str(*self)};
        if a == b { return ok(a); }
        do indent {
            match (ty::get(a).struct, ty::get(b).struct) {
              (ty::ty_bot, _) => {
                ok(a)
              }
              (ty::ty_var(a_id), ty::ty_var(b_id)) => {
                self.infcx().var_sub_var(&self.ty_var_bindings,
                                         a_id, b_id).then(|| ok(a) )
              }
              (ty::ty_var(a_id), _) => {
                self.infcx().var_sub_t(&self.ty_var_bindings,
                                       a_id, b).then(|| ok(a) )
              }
              (_, ty::ty_var(b_id)) => {
                self.infcx().t_sub_var(&self.ty_var_bindings,
                                       a, b_id).then(|| ok(a) )
              }
              (_, ty::ty_bot) => {
                err(ty::terr_sorts(b, a))
              }
              _ => {
                super_tys(&self, a, b)
              }
            }
        }
    }

    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty> {
        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        // (issue #2263).

        // First, we instantiate each bound region in the subtype with a fresh
        // region variable.
        let {fn_ty: a_fn_ty, _} = {
            do replace_bound_regions_in_fn_ty(self.tcx, @nil, none, a) |br| {
                // N.B.: The name of the bound region doesn't have
                // anything to do with the region variable that's created
                // for it.  The only thing we're doing with `br` here is
                // using it in the debug message.
                let rvar = self.infcx().next_region_var_nb();
                debug!{"Bound region %s maps to %s",
                       bound_region_to_str(self.tcx, br),
                       region_to_str(self.tcx, rvar)};
                rvar
            }
        };

        // Second, we instantiate each bound region in the supertype with a
        // fresh concrete region.
        let {fn_ty: b_fn_ty, _} = {
            do replace_bound_regions_in_fn_ty(self.tcx, @nil, none, b) |br| {
                // FIXME: eventually re_skolemized (issue #2263)
                ty::re_bound(br)
            }
        };

        // Try to compare the supertype and subtype now that they've been
        // instantiated.
        super_fns(&self, &a_fn_ty, &b_fn_ty)
    }

    // Traits please:

    fn flds(a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(&self, a, b)
    }

    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(&self, vk, a, b)
    }

    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode> {
        super_modes(&self, a, b)
    }

    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg> {
        super_args(&self, a, b)
    }

    fn substs(as: &ty::substs, bs: &ty::substs) -> cres<ty::substs> {
        super_substs(&self, as, bs)
    }

    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(&self, as, bs)
    }

    fn self_tys(a: option<ty::t>, b: option<ty::t>) -> cres<option<ty::t>> {
        super_self_tys(&self, a, b)
    }
}

impl lub: combine {
    fn infcx() -> infer_ctxt { *self }
    fn tag() -> ~str { ~"lub" }

    fn bot_ty(b: ty::t) -> cres<ty::t> { ok(b) }
    fn ty_bot(b: ty::t) -> cres<ty::t> { self.bot_ty(b) } // commutative

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx().tcx;

        debug!{"%s.mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b)};

        let m = if a.mutbl == b.mutbl {
            a.mutbl
        } else {
            m_const
        };

        match m {
          m_imm | m_const => {
            self.tys(a.ty, b.ty).chain(|t| ok({ty: t, mutbl: m}) )
          }

          m_mutbl => {
            self.infcx().try(|| {
                self.infcx().eq_tys(a.ty, b.ty).then(|| {
                    ok({ty: a.ty, mutbl: m})
                })
            }).chain_err(|_e| {
                self.tys(a.ty, b.ty).chain(|t| {
                    ok({ty: t, mutbl: m_const})
                })
            })
          }
        }
    }

    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        glb(self.infcx()).tys(a, b)
    }

    // XXX: Wrong.
    fn protos(p1: ty::fn_proto, p2: ty::fn_proto) -> cres<ty::fn_proto> {
        match (p1, p2) {
            (ty::proto_bare, _) => ok(p2),
            (_, ty::proto_bare) => ok(p1),
            (ty::proto_vstore(v1), ty::proto_vstore(v2)) => {
                self.infcx().try(|| {
                    do self.vstores(terr_fn, v1, v2).chain |vs| {
                        ok(ty::proto_vstore(vs))
                    }
                }).chain_err(|_err| {
                    // XXX: Totally unsound, but fixed up later.
                    ok(ty::proto_vstore(ty::vstore_slice(ty::re_static)))
                })
            }
        }
    }

    fn purities(f1: purity, f2: purity) -> cres<purity> {
        match (f1, f2) {
          (unsafe_fn, _) | (_, unsafe_fn) => ok(unsafe_fn),
          (impure_fn, _) | (_, impure_fn) => ok(impure_fn),
          (extern_fn, _) | (_, extern_fn) => ok(extern_fn),
          (pure_fn, pure_fn) => ok(pure_fn)
        }
    }

    fn ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style> {
        match (r1, r2) {
          (ast::return_val, _) |
          (_, ast::return_val) => ok(ast::return_val),
          (ast::noreturn, ast::noreturn) => ok(ast::noreturn)
        }
    }

    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region> {
        return glb(self.infcx()).regions(a, b);
    }

    fn regions(a: ty::region, b: ty::region) -> cres<ty::region> {
        debug!{"%s.regions(%?, %?)",
               self.tag(),
               a.to_str(self.infcx()),
               b.to_str(self.infcx())};

        do indent {
            match (a, b) {
              (ty::re_static, _) | (_, ty::re_static) => {
                ok(ty::re_static) // nothing lives longer than static
              }

              (ty::re_var(_), _) | (_, ty::re_var(_)) => {
                lattice_rvars(&self, a, b)
              }

              (f @ ty::re_free(f_id, _), ty::re_scope(s_id)) |
              (ty::re_scope(s_id), f @ ty::re_free(f_id, _)) => {
                // A "free" region can be interpreted as "some region
                // at least as big as the block f_id".  So, we can
                // reasonably compare free regions and scopes:
                let rm = self.infcx().tcx.region_map;
                match region::nearest_common_ancestor(rm, f_id, s_id) {
                  // if the free region's scope `f_id` is bigger than
                  // the scope region `s_id`, then the LUB is the free
                  // region itself:
                  some(r_id) if r_id == f_id => ok(f),

                  // otherwise, we don't know what the free region is,
                  // so we must conservatively say the LUB is static:
                  _ => ok(ty::re_static)
                }
              }

              (ty::re_scope(a_id), ty::re_scope(b_id)) => {
                // The region corresponding to an outer block is a
                // subtype of the region corresponding to an inner
                // block.
                let rm = self.infcx().tcx.region_map;
                match region::nearest_common_ancestor(rm, a_id, b_id) {
                  some(r_id) => ok(ty::re_scope(r_id)),
                  _ => ok(ty::re_static)
                }
              }

              // For these types, we cannot define any additional
              // relationship:
              (ty::re_free(_, _), ty::re_free(_, _)) |
              (ty::re_bound(_), ty::re_bound(_)) |
              (ty::re_bound(_), ty::re_free(_, _)) |
              (ty::re_bound(_), ty::re_scope(_)) |
              (ty::re_free(_, _), ty::re_bound(_)) |
              (ty::re_scope(_), ty::re_bound(_)) => {
                if a == b {
                    ok(a)
                } else {
                    ok(ty::re_static)
                }
              }
            }
        }
    }

    // Traits please:

    fn tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        lattice_tys(&self, a, b)
    }

    fn flds(a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(&self, a, b)
    }

    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(&self, vk, a, b)
    }

    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode> {
        super_modes(&self, a, b)
    }

    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg> {
        super_args(&self, a, b)
    }

    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty> {
        super_fns(&self, a, b)
    }

    fn substs(as: &ty::substs, bs: &ty::substs) -> cres<ty::substs> {
        super_substs(&self, as, bs)
    }

    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(&self, as, bs)
    }

    fn self_tys(a: option<ty::t>, b: option<ty::t>) -> cres<option<ty::t>> {
        super_self_tys(&self, a, b)
    }
}

impl glb: combine {
    fn infcx() -> infer_ctxt { *self }
    fn tag() -> ~str { ~"glb" }

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx().tcx;

        debug!{"%s.mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b)};

        match (a.mutbl, b.mutbl) {
          // If one side or both is mut, then the GLB must use
          // the precise type from the mut side.
          (m_mutbl, m_const) => {
            sub(*self).tys(a.ty, b.ty).chain(|_t| {
                ok({ty: a.ty, mutbl: m_mutbl})
            })
          }
          (m_const, m_mutbl) => {
            sub(*self).tys(b.ty, a.ty).chain(|_t| {
                ok({ty: b.ty, mutbl: m_mutbl})
            })
          }
          (m_mutbl, m_mutbl) => {
            self.infcx().eq_tys(a.ty, b.ty).then(|| {
                ok({ty: a.ty, mutbl: m_mutbl})
            })
          }

          // If one side or both is immutable, we can use the GLB of
          // both sides but mutbl must be `m_imm`.
          (m_imm, m_const) |
          (m_const, m_imm) |
          (m_imm, m_imm) => {
            self.tys(a.ty, b.ty).chain(|t| {
                ok({ty: t, mutbl: m_imm})
            })
          }

          // If both sides are const, then we can use GLB of both
          // sides and mutbl of only `m_const`.
          (m_const, m_const) => {
            self.tys(a.ty, b.ty).chain(|t| {
                ok({ty: t, mutbl: m_const})
            })
          }

          // There is no mutual subtype of these combinations.
          (m_mutbl, m_imm) |
          (m_imm, m_mutbl) => {
              err(ty::terr_mutability)
          }
        }
    }

    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        lub(self.infcx()).tys(a, b)
    }

    fn protos(p1: ty::fn_proto, p2: ty::fn_proto) -> cres<ty::fn_proto> {
        match (p1, p2) {
            (ty::proto_vstore(ty::vstore_slice(_)), _) => ok(p2),
            (_, ty::proto_vstore(ty::vstore_slice(_))) => ok(p1),
            (ty::proto_vstore(v1), ty::proto_vstore(v2)) => {
                self.infcx().try(|| {
                    do self.vstores(terr_fn, v1, v2).chain |vs| {
                        ok(ty::proto_vstore(vs))
                    }
                }).chain_err(|_err| {
                    // XXX: Totally unsound, but fixed up later.
                    ok(ty::proto_bare)
                })
            }
            _ => ok(ty::proto_bare)
        }
    }

    fn purities(f1: purity, f2: purity) -> cres<purity> {
        match (f1, f2) {
          (pure_fn, _) | (_, pure_fn) => ok(pure_fn),
          (extern_fn, _) | (_, extern_fn) => ok(extern_fn),
          (impure_fn, _) | (_, impure_fn) => ok(impure_fn),
          (unsafe_fn, unsafe_fn) => ok(unsafe_fn)
        }
    }

    fn ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style> {
        match (r1, r2) {
          (ast::return_val, ast::return_val) => {
            ok(ast::return_val)
          }
          (ast::noreturn, _) |
          (_, ast::noreturn) => {
            ok(ast::noreturn)
          }
        }
    }

    fn regions(a: ty::region, b: ty::region) -> cres<ty::region> {
        debug!{"%s.regions(%?, %?)",
               self.tag(),
               a.to_str(self.infcx()),
               b.to_str(self.infcx())};

        do indent {
            match (a, b) {
              (ty::re_static, r) | (r, ty::re_static) => {
                // static lives longer than everything else
                ok(r)
              }

              (ty::re_var(_), _) | (_, ty::re_var(_)) => {
                lattice_rvars(&self, a, b)
              }

              (ty::re_free(f_id, _), s @ ty::re_scope(s_id)) |
              (s @ ty::re_scope(s_id), ty::re_free(f_id, _)) => {
                // Free region is something "at least as big as
                // `f_id`."  If we find that the scope `f_id` is bigger
                // than the scope `s_id`, then we can say that the GLB
                // is the scope `s_id`.  Otherwise, as we do not know
                // big the free region is precisely, the GLB is undefined.
                let rm = self.infcx().tcx.region_map;
                match region::nearest_common_ancestor(rm, f_id, s_id) {
                  some(r_id) if r_id == f_id => ok(s),
                  _ => err(ty::terr_regions_no_overlap(b, a))
                }
              }

              (ty::re_scope(a_id), ty::re_scope(b_id)) |
              (ty::re_free(a_id, _), ty::re_free(b_id, _)) => {
                if a == b {
                    // Same scope or same free identifier, easy case.
                    ok(a)
                } else {
                    // We want to generate the intersection of two
                    // scopes or two free regions.  So, if one of
                    // these scopes is a subscope of the other, return
                    // it.  Otherwise fail.
                    let rm = self.infcx().tcx.region_map;
                    match region::nearest_common_ancestor(rm, a_id, b_id) {
                      some(r_id) if a_id == r_id => ok(ty::re_scope(b_id)),
                      some(r_id) if b_id == r_id => ok(ty::re_scope(a_id)),
                      _ => err(ty::terr_regions_no_overlap(b, a))
                    }
                }
              }

              // For these types, we cannot define any additional
              // relationship:
              (ty::re_bound(_), ty::re_bound(_)) |
              (ty::re_bound(_), ty::re_free(_, _)) |
              (ty::re_bound(_), ty::re_scope(_)) |
              (ty::re_free(_, _), ty::re_bound(_)) |
              (ty::re_scope(_), ty::re_bound(_)) => {
                if a == b {
                    ok(a)
                } else {
                    err(ty::terr_regions_no_overlap(b, a))
                }
              }
            }
        }
    }

    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region> {
        lub(self.infcx()).regions(a, b)
    }

    fn tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        lattice_tys(&self, a, b)
    }

    // Traits please:

    fn flds(a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(&self, a, b)
    }

    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(&self, vk, a, b)
    }

    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode> {
        super_modes(&self, a, b)
    }

    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg> {
        super_args(&self, a, b)
    }

    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty> {
        super_fns(&self, a, b)
    }

    fn substs(as: &ty::substs, bs: &ty::substs) -> cres<ty::substs> {
        super_substs(&self, as, bs)
    }

    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(&self, as, bs)
    }

    fn self_tys(a: option<ty::t>, b: option<ty::t>) -> cres<option<ty::t>> {
        super_self_tys(&self, a, b)
    }
}

// ______________________________________________________________________
// Lattice operations on variables
//
// This is common code used by both LUB and GLB to compute the LUB/GLB
// for pairs of variables or for variables and values.

trait lattice_ops {
    fn bnd<T:copy>(b: bounds<T>) -> option<T>;
    fn with_bnd<T:copy>(b: bounds<T>, t: T) -> bounds<T>;
    fn ty_bot(t: ty::t) -> cres<ty::t>;
}

impl lub: lattice_ops {
    fn bnd<T:copy>(b: bounds<T>) -> option<T> { b.ub }
    fn with_bnd<T:copy>(b: bounds<T>, t: T) -> bounds<T> {
        {ub: some(t) with b}
    }
    fn ty_bot(t: ty::t) -> cres<ty::t> {
        ok(t)
    }
}

impl glb: lattice_ops {
    fn bnd<T:copy>(b: bounds<T>) -> option<T> { b.lb }
    fn with_bnd<T:copy>(b: bounds<T>, t: T) -> bounds<T> {
        {lb: some(t) with b}
    }
    fn ty_bot(_t: ty::t) -> cres<ty::t> {
        ok(ty::mk_bot(self.infcx().tcx))
    }
}

fn lattice_tys<L:lattice_ops combine>(
    self: &L, a: ty::t, b: ty::t) -> cres<ty::t> {

    debug!{"%s.lattice_tys(%s, %s)", self.tag(),
           a.to_str(self.infcx()),
           b.to_str(self.infcx())};
    if a == b { return ok(a); }
    do indent {
        match (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_bot, _) => self.ty_bot(b),
          (_, ty::ty_bot) => self.ty_bot(a),

          (ty::ty_var(a_id), ty::ty_var(b_id)) => {
            lattice_vars(self, &self.infcx().ty_var_bindings,
                         a, a_id, b_id,
                         |x, y| self.tys(x, y) )
          }

          (ty::ty_var(a_id), _) => {
            lattice_var_and_t(self, &self.infcx().ty_var_bindings, a_id, b,
                              |x, y| self.tys(x, y) )
          }

          (_, ty::ty_var(b_id)) => {
            lattice_var_and_t(self, &self.infcx().ty_var_bindings, b_id, a,
                              |x, y| self.tys(x, y) )
          }
          _ => {
            super_tys(self, a, b)
          }
        }
    }
}

// Pull out some common code from LUB/GLB for handling region vars:
fn lattice_rvars<L:lattice_ops combine>(
    self: &L, a: ty::region, b: ty::region) -> cres<ty::region> {

    match (a, b) {
      (ty::re_var(a_id), ty::re_var(b_id)) => {
        lattice_vars(self, &self.infcx().region_var_bindings,
                     a, a_id, b_id,
                     |x, y| self.regions(x, y) )
      }

      (ty::re_var(v_id), r) | (r, ty::re_var(v_id)) => {
        lattice_var_and_t(self, &self.infcx().region_var_bindings,
                          v_id, r,
                          |x, y| self.regions(x, y) )
      }

      _ => {
        self.infcx().tcx.sess.bug(
            fmt!{"%s: lattice_rvars invoked with a=%s and b=%s, \
                  neither of which are region variables",
                 self.tag(),
                 a.to_str(self.infcx()),
                 b.to_str(self.infcx())});
      }
    }
}

fn lattice_vars<V:copy vid, T:copy to_str st, L:lattice_ops combine>(
    self: &L, vb: &vals_and_bindings<V, bounds<T>>,
    +a_t: T, +a_vid: V, +b_vid: V,
    c_ts: fn(T, T) -> cres<T>) -> cres<T> {

    // The comments in this function are written for LUB and types,
    // but they apply equally well to GLB and regions if you inverse
    // upper/lower/sub/super/etc.

    // Need to find a type that is a supertype of both a and b:
    let nde_a = self.infcx().get(vb, a_vid);
    let nde_b = self.infcx().get(vb, b_vid);
    let a_vid = nde_a.root;
    let b_vid = nde_b.root;
    let a_bounds = nde_a.possible_types;
    let b_bounds = nde_b.possible_types;

    debug!{"%s.lattice_vars(%s=%s <: %s=%s)",
           self.tag(),
           a_vid.to_str(), a_bounds.to_str(self.infcx()),
           b_vid.to_str(), b_bounds.to_str(self.infcx())};

    if a_vid == b_vid {
        return ok(a_t);
    }

    // If both A and B have an UB type, then we can just compute the
    // LUB of those types:
    let a_bnd = self.bnd(a_bounds), b_bnd = self.bnd(b_bounds);
    match (a_bnd, b_bnd) {
      (some(a_ty), some(b_ty)) => {
        match self.infcx().try(|| c_ts(a_ty, b_ty) ) {
            ok(t) => return ok(t),
            err(_) => { /*fallthrough */ }
        }
      }
      _ => {/*fallthrough*/}
    }

    // Otherwise, we need to merge A and B into one variable.  We can
    // then use either variable as an upper bound:
    self.infcx().var_sub_var(vb, a_vid, b_vid).then(|| ok(a_t) )
}

fn lattice_var_and_t<V:copy vid, T:copy to_str st, L:lattice_ops combine>(
    self: &L, vb: &vals_and_bindings<V, bounds<T>>,
    +a_id: V, +b: T,
    c_ts: fn(T, T) -> cres<T>) -> cres<T> {

    let nde_a = self.infcx().get(vb, a_id);
    let a_id = nde_a.root;
    let a_bounds = nde_a.possible_types;

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    debug!{"%s.lattice_var_and_t(%s=%s <: %s)",
           self.tag(),
           a_id.to_str(), a_bounds.to_str(self.infcx()),
           b.to_str(self.infcx())};

    match self.bnd(a_bounds) {
      some(a_bnd) => {
        // If a has an upper bound, return the LUB(a.ub, b)
        debug!{"bnd=some(%s)", a_bnd.to_str(self.infcx())};
        return c_ts(a_bnd, b);
      }
      none => {
        // If a does not have an upper bound, make b the upper bound of a
        // and then return b.
        debug!{"bnd=none"};
        let a_bounds = self.with_bnd(a_bounds, b);
        do self.infcx().bnds(a_bounds.lb, a_bounds.ub).then {
            self.infcx().set(vb, a_id, root(a_bounds,
                                            nde_a.rank));
            ok(b)
        }
      }
    }
}
