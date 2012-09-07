/*!

Region inference module.

# Introduction

Region inference uses a somewhat more involved algorithm than type
inference.  It is not the most efficient thing ever written though it
seems to work well enough in practice (famous last words).  The reason
that we use a different algorithm is because, unlike with types, it is
impractical to hand-annotate with regions (in some cases, there aren't
even the requisite syntactic forms).  So we have to get it right, and
it's worth spending more time on a more involved analysis.  Moreover,
regions are a simpler case than types: they don't have aggregate
structure, for example.

Unlike normal type inference, which is similar in spirit H-M and thus
works progressively, the region type inference works by accumulating
constraints over the course of a function.  Finally, at the end of
processing a function, we process and solve the constraints all at
once.

The constraints are always of one of three possible forms:

- ConstrainVarSubVar(R_i, R_j) states that region variable R_i
  must be a subregion of R_j
- ConstrainRegSubVar(R, R_i) states that the concrete region R
  (which must not be a variable) must be a subregion of the varibale R_i
- ConstrainVarSubReg(R_i, R) is the inverse

# Building up the constraints

Variables and constraints are created using the following methods:

- `new_region_var()` creates a new, unconstrained region variable;
- `make_subregion(R_i, R_j)` states that R_i is a subregion of R_j
- `lub_regions(R_i, R_j) -> R_k` returns a region R_k which is
  the smallest region that is greater than both R_i and R_j
- `glb_regions(R_i, R_j) -> R_k` returns a region R_k which is
  the greatest region that is smaller than both R_i and R_j

The actual region resolution algorithm is not entirely
obvious, though it is also not overly complex.  I'll explain
the algorithm as it currently works, then explain a somewhat
more complex variant that would probably scale better for
large graphs (and possibly all graphs).

## Snapshotting

It is also permitted to try (and rollback) changes to the graph.  This
is done by invoking `start_snapshot()`, which returns a value.  Then
later you can call `rollback_to()` which undoes the work.
Alternatively, you can call `commit()` which ends all snapshots.
Snapshots can be recursive---so you can start a snapshot when another
is in progress, but only the root snapshot can "commit".

# Resolving constraints

The constraint resolution algorithm is not super complex but also not
entirely obvious.  Here I describe the problem somewhat abstractly,
then describe how the current code works, and finally describe a
better solution that is as of yet unimplemented.  There may be other,
smarter ways of doing this with which I am unfamiliar and can't be
bothered to research at the moment. - NDM

## The problem

Basically our input is a directed graph where nodes can be divided
into two categories: region variables and concrete regions.  Each edge
`R -> S` in the graph represents a constraint that the region `R` is a
subregion of the region `S`.

Region variable nodes can have arbitrary degree.  There is one region
variable node per region variable.

Each concrete region node is associated with some, well, concrete
region: e.g., a free lifetime, or the region for a particular scope.
Note that there may be more than one concrete region node for a
particular region value.  Moreover, because of how the graph is built,
we know that all concrete region nodes have either in-degree 1 or
out-degree 1.

Before resolution begins, we build up the constraints in a hashmap
that maps `Constraint` keys to spans.  During resolution, we construct
the actual `Graph` structure that we describe here.

## Our current algorithm

We divide region variables into two groups: Expanding and Contracting.
Expanding region variables are those that have a concrete region
predecessor (direct or indirect).  Contracting region variables are
all others.

We first resolve the values of Expanding region variables and then
process Contracting ones.  We currently use an iterative, fixed-point
procedure (but read on, I believe this could be replaced with a linear
walk).  Basically we iterate over the edges in the graph, ensuring
that, if the source of the edge has a value, then this value is a
subregion of the target value.  If the target does not yet have a
value, it takes the value from the source.  If the target already had
a value, then the resulting value is Least Upper Bound of the old and
new values. When we are done, each Expanding node will have the
smallest region that it could possibly have and still satisfy the
constraints.

We next process the Contracting nodes.  Here we again iterate over the
edges, only this time we move values from target to source (if the
source is a Contracting node).  For each contracting node, we compute
its value as the GLB of all its successors.  Basically contracting
nodes ensure that there is overlap between their successors; we will
ultimately infer the largest overlap possible.

### A better algorithm

Fixed-point iteration is not necessary.  What we ought to do is first
identify and remove strongly connected components (SCC) in the graph.
Note that such components must consist solely of region variables; all
of these variables can effectively be unified into a single variable.

Once SCCs are removed, we are left with a DAG.  At this point, we can
walk the DAG in toplogical order once to compute the expanding nodes,
and again in reverse topological order to compute the contracting
nodes.The main reason I did not write it this way is that I did not
feel like implementing the SCC and toplogical sort algorithms at the
moment.

# Skolemization and functions

One of the trickiest and most subtle aspects of regions is dealing
with the fact that region variables are bound in function types.  I
strongly suggest that if you want to understand the situation, you
read this paper (which is, admittedly, very long, but you don't have
to read the whole thing):

http://research.microsoft.com/en-us/um/people/simonpj/papers/higher-rank/

NOTE--for the most part, we do not yet handle these cases correctly!

## Subtyping and bound regions

### Simple examples

The situation is well-summarized by these examples (here I am omitting
the types as they are not interesting, and I am writing binding
explicitly):

    1. fn<a>(&a/T) <: fn<b>(&b/T)?        Yes: a -> b
    2. fn<a>(&a/T) <: fn(&b/T)?           Yes: a -> b
    3. fn(&a/T)    <: fn<b>(&b/T)?        No!
    4. fn(&a/T)    <: fn(&b/T)?           No!
    5. fn(&a/T)    <: fn(&a)?           Yes!

In case one, the two function types are equivalent because both
reference a bound region, just with different names.

In case two, the subtyping relationship is valid because the subtyping
function accepts a pointer in *any* region, whereas the supertype
function accepts a pointer *only in the region `b`*.  Therefore, it is
safe to use the subtype wherever the supertype is expected, as the
supertype can only be passed pointers in region `b`, and the subtype
can handle `b` (but also others).

Case three is the opposite: here the subtype requires the region `a`,
but the supertype must accept pointers in any region.  That means that
it is not safe to use the subtype where the supertype is expected: the
supertype can be passed pointers in any region, but the subtype can
only handle pointers in the region `a`.

Case four is fairly simple.  The subtype expects region `a` but the supertype
expects region `b`.  These two regions are not the same.  Therefore, not
a subtype.

Case five is similar to four, except that the subtype and supertype
expect the same region, so in fact they are the same type.  That's
fine.

Here is the algorithm we use to perform the subtyping check:

1. Replace all bound regions in the subtype with new variables
2. Replace all bound regions in the supertype with skolemized
   equivalents.  A "skolemized" region is just a new fresh region
   name.
3. Check that the parameter and return types match as normal
4. Ensure that no skolemized regions 'leak' into region variables
   visible from "the outside"

I'll walk briefly through how this works with the examples above.
I'll ignore the last step for now, it'll come up in the complex
examples below.

#### First example

Let's look first at the first example, which was:

    1. fn<a>(&a/T) <: fn<b>(&b/T/T)?        Yes: a -> x

After steps 1 and 2 of the algorithm we will have replaced the types
like so:

    1. fn(&A/T) <: fn(&x/T)?

Here the upper case `&A` indicates a *region variable*, that is, a
region whose value is being inferred by the system.  I also replaced
`&b` with `&x`---I'll use letters late in the alphabet (`x`, `y`, `z`)
to indicate skolemized region names.  We can assume they don't appear
elsewhere.  Note that neither the sub- nor the supertype bind any
region names anymore (that is, the `<a>` and `<b>` have been removed).

The next step is to check that the parameter types match.  Because
parameters are contravariant, this means that we check whether:

    &x/T <: &A/T

Region pointers are contravariant so this implies that

    &A <= &x

must hold, where `<=` is the subregion relationship.  Processing
*this* constrain simply adds a constraint into our graph that `&A <=
&x` and is considered successful (it can, for example, be satisfied by
choosing the value `&x` for `&A`).

So far we have encountered no error, so the subtype check succeeds.

#### The third example

Now let's look first at the third example, which was:

    3. fn(&a/T)    <: fn<b>(&b/T)?        No!

After steps 1 and 2 of the algorithm we will have replaced the types
like so:

    3. fn(&a/T) <: fn(&x/T)?

This looks pretty much the same as before, except that on the LHS `&a`
was not bound, and hence was left as-is and not replaced with a
variable.  The next step is again to check that the parameter types
match.  This will ultimately require (as before) that `&a` <= `&x`
must hold: but this does not hold.  `a` and `x` are both distinct free
regions.  So the subtype check fails.

#### Checking for skolemization leaks

You may be wondering about that mysterious last step.  So far it has not
been relevant.  The purpose of that last step is to catch something like
*this*:

    fn<a>() -> fn(&a/T) <: fn() -> fn<b>(&b/T)?   No.

Here the function types are the same but for where the binding occurs.
The subtype returns a function that expects a value in precisely one
region.  The supertype returns a function that expects a value in any
region.  If we allow an instance of the subtype to be used where the
supertype is expected, then, someone could call the fn and think that
the return value has type `fn<b>(&b/T)` when it really has type
`fn(&a/T)` (this is case #3, above).  Bad.

So let's step through what happens when we perform this subtype check.
We first replace the bound regions in the subtype (the supertype has
no bound regions).  This gives us:

    fn() -> fn(&A/T) <: fn() -> fn<b>(&b/T)?

Now we compare the return types, which are covariant, and hence we have:

    fn(&A/T) <: fn<b>(&b/T)?

Here we skolemize the bound region in the supertype to yield:

    fn(&A/T) <: fn(&x/T)?

And then proceed to compare the argument types:

    &x/T <: &A/T
    &A <= &x

Finally, this is where it gets interesting!  This is where an error
*should* be reported.  But in fact this will not happen.  The reason why
is that `A` is a variable: we will infer that its value is the fresh
region `x` and think that everything is happy.  In fact, this behavior
is *necessary*, it was key to the first example we walked through.

The difference between this example and the first one is that the variable
`A` already existed at the point where the skolemization occurred.  In
the first example, you had two functions:

    fn<a>(&a/T) <: fn<b>(&b/T)

and hence `&A` and `&x` were created "together".  In general, the
intention of the skolemized names is that they are supposed to be
fresh names that could never be equal to anything from the outside.
But when inference comes into play, we might not be respecting this
rule.

So the way we solve this is to add a fourth step that examines the
constraints that refer to skolemized names.  Basically, consider a
non-directed verison of the constraint graph.  The only things
reachable from a skolemized region ought to be the region variables
that were created at the same time.  So this case here would fail
because `&x` was created alone, but is relatable to `&A`.

*/

#[warn(deprecated_mode)];
#[warn(deprecated_pattern)];

use dvec::DVec;
use result::Result;
use result::{Ok, Err};
use std::map::{hashmap, uint_hash};
use std::cell::{Cell, empty_cell};
use std::list::{List, Nil, Cons};

use ty::{region, RegionVid, hash_region};
use region::is_subregion_of;
use syntax::codemap;
use to_str::to_str;
use util::ppaux::note_and_explain_region;

export RegionVarBindings;
export make_subregion;
export lub_regions;
export glb_regions;

enum Constraint {
    ConstrainVarSubVar(RegionVid, RegionVid),
    ConstrainRegSubVar(region, RegionVid),
    ConstrainVarSubReg(RegionVid, region)
}

impl Constraint: cmp::Eq {
    pure fn eq(&&other: Constraint) -> bool {
        match (self, other) {
            (ConstrainVarSubVar(v0a, v1a), ConstrainVarSubVar(v0b, v1b)) => {
                v0a == v0b && v1a == v1b
            }
            (ConstrainRegSubVar(ra, va), ConstrainRegSubVar(rb, vb)) => {
                ra == rb && va == vb
            }
            (ConstrainVarSubReg(va, ra), ConstrainVarSubReg(vb, rb)) => {
                va == vb && ra == rb
            }
            (ConstrainVarSubVar(*), _) => false,
            (ConstrainRegSubVar(*), _) => false,
            (ConstrainVarSubReg(*), _) => false
        }
    }
    pure fn ne(&&other: Constraint) -> bool { !self.eq(other) }
}

struct TwoRegions {
    a: region;
    b: region;
}

impl TwoRegions: cmp::Eq {
    pure fn eq(&&other: TwoRegions) -> bool {
        self.a == other.a && self.b == other.b
    }
    pure fn ne(&&other: TwoRegions) -> bool { !self.eq(other) }
}

enum UndoLogEntry {
    Snapshot,
    AddVar(RegionVid),
    AddConstraint(Constraint),
    AddCombination(CombineMap, TwoRegions)
}

type CombineMap = hashmap<TwoRegions, RegionVid>;

struct RegionVarBindings {
    tcx: ty::ctxt;
    var_spans: DVec<span>;
    values: Cell<~[ty::region]>;
    constraints: hashmap<Constraint, span>;
    lubs: CombineMap;
    glbs: CombineMap;

    // The undo log records actions that might later be undone.
    //
    // Note: when the undo_log is empty, we are not actively
    // snapshotting.  When the `start_snapshot()` method is called, we
    // push a Snapshot entry onto the list to indicate that we are now
    // actively snapshotting.  The reason for this is that otherwise
    // we end up adding entries for things like the lower bound on
    // a variable and so forth, which can never be rolled back.
    undo_log: DVec<UndoLogEntry>;
}

fn RegionVarBindings(tcx: ty::ctxt) -> RegionVarBindings {
    RegionVarBindings {
        tcx: tcx,
        var_spans: DVec(),
        values: empty_cell(),
        constraints: hashmap(hash_constraint, sys::shape_eq),
        lubs: CombineMap(),
        glbs: CombineMap(),
        undo_log: DVec()
    }
}

// Note: takes two regions but doesn't care which is `a` and which is
// `b`!  Not obvious that this is the most efficient way to go about
// it.
fn CombineMap() -> CombineMap {
    return hashmap(hash_two_regions, eq_two_regions);

    pure fn hash_two_regions(rc: &TwoRegions) -> uint {
        hash_region(&rc.a) ^ hash_region(&rc.b)
    }

    pure fn eq_two_regions(rc1: &TwoRegions, rc2: &TwoRegions) -> bool {
        (rc1.a == rc2.a && rc1.b == rc2.b) ||
            (rc1.a == rc2.b && rc1.b == rc2.a)
    }
}

pure fn hash_constraint(rc: &Constraint) -> uint {
    match *rc {
      ConstrainVarSubVar(a, b) => *a ^ *b,
      ConstrainRegSubVar(ref r, b) => ty::hash_region(r) ^ *b,
      ConstrainVarSubReg(a, ref r) => *a ^ ty::hash_region(r)
    }
}

impl RegionVarBindings {
    fn in_snapshot() -> bool {
        self.undo_log.len() > 0
    }

    fn start_snapshot() -> uint {
        debug!("RegionVarBindings: snapshot()=%u", self.undo_log.len());
        if self.in_snapshot() {
            self.undo_log.len()
        } else {
            self.undo_log.push(Snapshot);
            0
        }
    }

    fn commit() {
        debug!("RegionVarBindings: commit()");
        while self.undo_log.len() > 0 {
            self.undo_log.pop();
        }
    }

    fn rollback_to(snapshot: uint) {
        debug!("RegionVarBindings: rollback_to(%u)", snapshot);
        while self.undo_log.len() > snapshot {
            let undo_item = self.undo_log.pop();
            debug!("undo_item=%?", undo_item);
            match undo_item {
              Snapshot => {}
              AddVar(vid) => {
                assert self.var_spans.len() == *vid + 1;
                self.var_spans.pop();
              }
              AddConstraint(constraint) => {
                self.constraints.remove(constraint);
              }
              AddCombination(map, regions) => {
                map.remove(regions);
              }
            }
        }
    }

    fn num_vars() -> uint {
        self.var_spans.len()
    }

    fn new_region_var(span: span) -> RegionVid {
        let id = self.num_vars();
        self.var_spans.push(span);
        let vid = RegionVid(id);
        if self.in_snapshot() {
            self.undo_log.push(AddVar(vid));
        }
        debug!("created new region variable %? with span %?",
               vid, codemap::span_to_str(span, self.tcx.sess.codemap));
        return vid;
    }

    fn add_constraint(+constraint: Constraint, span: span) {
        // cannot add constraints once regions are resolved
        assert self.values.is_empty();

        debug!("RegionVarBindings: add_constraint(%?)", constraint);

        if self.constraints.insert(constraint, span) {
            if self.in_snapshot() {
                self.undo_log.push(AddConstraint(constraint));
            }
        }
    }

    fn make_subregion(span: span, sub: region, sup: region) -> cres<()> {
        // cannot add constraints once regions are resolved
        assert self.values.is_empty();

        debug!("RegionVarBindings: make_subregion(%?, %?)", sub, sup);
        match (sub, sup) {
          (ty::re_var (sub_id), ty::re_var(sup_id)) => {
            self.add_constraint(ConstrainVarSubVar(sub_id, sup_id), span);
            Ok(())
          }
          (r, ty::re_var(sup_id)) => {
            self.add_constraint(ConstrainRegSubVar(r, sup_id), span);
            Ok(())
          }
          (ty::re_var(sub_id), r) => {
            self.add_constraint(ConstrainVarSubReg(sub_id, r), span);
            Ok(())
          }
          _ => {
            if self.is_subregion_of(sub, sup) {
                Ok(())
            } else {
                Err(ty::terr_regions_does_not_outlive(sub, sup))
            }
          }
        }
    }

    fn lub_regions(span: span, a: region, b: region) -> cres<region> {
        // cannot add constraints once regions are resolved
        assert self.values.is_empty();

        debug!("RegionVarBindings: lub_regions(%?, %?)", a, b);
        match (a, b) {
          (ty::re_static, _) | (_, ty::re_static) => {
            Ok(ty::re_static) // nothing lives longer than static
          }

          (ty::re_var(*), _) | (_, ty::re_var(*)) => {
            self.combine_vars(
                self.lubs, a, b, span,
                |old_r, new_r| self.make_subregion(span, old_r, new_r))
          }

          _ => {
            Ok(self.lub_concrete_regions(a, b))
          }
        }
    }

    fn glb_regions(span: span, a: region, b: region) -> cres<region> {
        // cannot add constraints once regions are resolved
        assert self.values.is_empty();

        debug!("RegionVarBindings: glb_regions(%?, %?)", a, b);
        match (a, b) {
          (ty::re_static, r) | (r, ty::re_static) => {
            // static lives longer than everything else
            Ok(r)
          }

          (ty::re_var(*), _) | (_, ty::re_var(*)) => {
            self.combine_vars(
                self.glbs, a, b, span,
                |old_r, new_r| self.make_subregion(span, new_r, old_r))
          }

          _ => {
            self.glb_concrete_regions(a, b)
          }
        }
    }

    fn resolve_var(rid: RegionVid) -> ty::region {
        debug!("RegionVarBindings: resolve_var(%?)", rid);
        if self.values.is_empty() {
            self.tcx.sess.span_bug(
                self.var_spans[*rid],
                fmt!("Attempt to resolve region variable before values have \
                      been computed!"));
        }

        self.values.with_ref(|values| values[*rid])
    }

    fn combine_vars(combines: CombineMap, a: region, b: region, span: span,
                    relate: fn(old_r: region, new_r: region) -> cres<()>)
        -> cres<region> {

        let vars = TwoRegions { a: a, b: b };
        match combines.find(vars) {
          Some(c) => Ok(ty::re_var(c)),
          None => {
            let c = self.new_region_var(span);
            combines.insert(vars, c);
            if self.in_snapshot() {
                self.undo_log.push(AddCombination(combines, vars));
            }
            do relate(a, ty::re_var(c)).then {
                do relate(b, ty::re_var(c)).then {
                    debug!("combine_vars() c=%?", ty::re_var(c));
                    Ok(ty::re_var(c))
                }
            }
          }
        }
    }

    /**
    This function performs the actual region resolution.  It must be
    called after all constraints have been added.  It performs a
    fixed-point iteration to find region values which satisfy all
    constraints, assuming such values can be found; if they cannot,
    errors are reported.
    */
    fn resolve_regions() {
        debug!("RegionVarBindings: resolve_regions()");
        self.values.put_back(self.infer_variable_values());
    }
}

priv impl RegionVarBindings {
    fn is_subregion_of(sub: region, sup: region) -> bool {
        is_subregion_of(self.tcx.region_map, sub, sup)
    }

    fn lub_concrete_regions(+a: region, +b: region) -> region {
        match (a, b) {
          (ty::re_static, _) | (_, ty::re_static) => {
            ty::re_static // nothing lives longer than static
          }

          (ty::re_var(v_id), _) | (_, ty::re_var(v_id)) => {
            self.tcx.sess.span_bug(
                self.var_spans[*v_id],
                fmt!("lub_concrete_regions invoked with \
                      non-concrete regions: %?, %?", a, b));
          }

          (f @ ty::re_free(f_id, _), ty::re_scope(s_id)) |
          (ty::re_scope(s_id), f @ ty::re_free(f_id, _)) => {
            // A "free" region can be interpreted as "some region
            // at least as big as the block f_id".  So, we can
            // reasonably compare free regions and scopes:
            let rm = self.tcx.region_map;
            match region::nearest_common_ancestor(rm, f_id, s_id) {
              // if the free region's scope `f_id` is bigger than
              // the scope region `s_id`, then the LUB is the free
              // region itself:
              Some(r_id) if r_id == f_id => f,

              // otherwise, we don't know what the free region is,
              // so we must conservatively say the LUB is static:
              _ => ty::re_static
            }
          }

          (ty::re_scope(a_id), ty::re_scope(b_id)) => {
            // The region corresponding to an outer block is a
            // subtype of the region corresponding to an inner
            // block.
            let rm = self.tcx.region_map;
            match region::nearest_common_ancestor(rm, a_id, b_id) {
              Some(r_id) => ty::re_scope(r_id),
              _ => ty::re_static
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
            if a == b {a} else {ty::re_static}
          }
        }
    }

    fn glb_concrete_regions(+a: region, +b: region) -> cres<region> {
        match (a, b) {
          (ty::re_static, r) | (r, ty::re_static) => {
            // static lives longer than everything else
            Ok(r)
          }

          (ty::re_var(v_id), _) | (_, ty::re_var(v_id)) => {
            self.tcx.sess.span_bug(
                self.var_spans[*v_id],
                fmt!("glb_concrete_regions invoked with \
                      non-concrete regions: %?, %?", a, b));
          }

          (ty::re_free(f_id, _), s @ ty::re_scope(s_id)) |
          (s @ ty::re_scope(s_id), ty::re_free(f_id, _)) => {
            // Free region is something "at least as big as
            // `f_id`."  If we find that the scope `f_id` is bigger
            // than the scope `s_id`, then we can say that the GLB
            // is the scope `s_id`.  Otherwise, as we do not know
            // big the free region is precisely, the GLB is undefined.
            let rm = self.tcx.region_map;
            match region::nearest_common_ancestor(rm, f_id, s_id) {
              Some(r_id) if r_id == f_id => Ok(s),
              _ => Err(ty::terr_regions_no_overlap(b, a))
            }
          }

          (ty::re_scope(a_id), ty::re_scope(b_id)) |
          (ty::re_free(a_id, _), ty::re_free(b_id, _)) => {
            if a == b {
                // Same scope or same free identifier, easy case.
                Ok(a)
            } else {
                // We want to generate the intersection of two
                // scopes or two free regions.  So, if one of
                // these scopes is a subscope of the other, return
                // it.  Otherwise fail.
                let rm = self.tcx.region_map;
                match region::nearest_common_ancestor(rm, a_id, b_id) {
                  Some(r_id) if a_id == r_id => Ok(ty::re_scope(b_id)),
                  Some(r_id) if b_id == r_id => Ok(ty::re_scope(a_id)),
                  _ => Err(ty::terr_regions_no_overlap(b, a))
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
                Ok(a)
            } else {
                Err(ty::terr_regions_no_overlap(b, a))
            }
          }
        }
    }

    fn report_type_error(span: span, terr: &ty::type_err) {
        let terr_str = ty::type_err_to_str(self.tcx, terr);
        self.tcx.sess.span_err(span, terr_str);
    }
}

// ______________________________________________________________________

enum Direction { Incoming = 0, Outgoing = 1 }

impl Direction : cmp::Eq {
    pure fn eq(&&other: Direction) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: Direction) -> bool { !self.eq(other) }
}

enum Classification { Expanding, Contracting }

impl Classification : cmp::Eq {
    pure fn eq(&&other: Classification) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: Classification) -> bool { !self.eq(other) }
}

enum GraphNodeValue { NoValue, Value(region), ErrorValue }

struct GraphNode {
    span: span;
    mut classification: Classification;
    mut value: GraphNodeValue;
    head_edge: [mut uint * 2]; // FIXME(#3226)--should not need mut
}

struct GraphEdge {
    next_edge: [mut uint * 2]; // FIXME(#3226)--should not need mut
    constraint: Constraint;
    span: span;
}

struct Graph {
    nodes: ~[GraphNode];
    edges: ~[GraphEdge];
}

struct SpannedRegion {
    region: region;
    span: span;
}

type TwoRegionsMap = hashmap<TwoRegions, ()>;

fn TwoRegionsMap() -> TwoRegionsMap {
    return hashmap(hash_two_regions, sys::shape_eq);

    pure fn hash_two_regions(rc: &TwoRegions) -> uint {
        hash_region(&rc.a) ^ (hash_region(&rc.b) << 2)
    }
}

impl RegionVarBindings {
    fn infer_variable_values() -> ~[region] {
        let graph = self.construct_graph();
        self.expansion(&graph);
        self.contraction(&graph);
        self.extract_regions_and_report_errors(&graph)
    }

    fn construct_graph() -> Graph {
        let num_vars = self.num_vars();
        let num_edges = self.constraints.size();

        let nodes = vec::from_fn(num_vars, |var_idx| {
            GraphNode {
                // All nodes are initially classified as contracting; during
                // the expansion phase, we will shift the classification for
                // those nodes that have a concrete region predecessor to
                // Expanding.
                classification: Contracting,
                span: self.var_spans[var_idx],
                value: NoValue,
                head_edge: [mut uint::max_value, uint::max_value]
            }
        });

        // It would be nice to write this using map():
        let mut edges = ~[];
        vec::reserve(edges, num_edges);
        for self.constraints.each_ref |constraint, span| {
            vec::push(edges, GraphEdge {
                next_edge: [mut uint::max_value, uint::max_value],
                constraint: *constraint,
                span: *span
            });
        }

        let mut graph = Graph {
            nodes: move nodes,
            edges: move edges
        };

        for uint::range(0, num_edges) |edge_idx| {
            match graph.edges[edge_idx].constraint {
              ConstrainVarSubVar(copy a_id, copy b_id) => {
                insert_edge(&mut graph, a_id, Outgoing, edge_idx);
                insert_edge(&mut graph, b_id, Incoming, edge_idx);
              }
              ConstrainRegSubVar(_, copy b_id) => {
                insert_edge(&mut graph, b_id, Incoming, edge_idx);
              }
              ConstrainVarSubReg(copy a_id, _) => {
                insert_edge(&mut graph, a_id, Outgoing, edge_idx);
              }
            }
        }

        return graph;

        fn insert_edge(graph: &mut Graph,
                       node_id: RegionVid,
                       edge_dir: Direction,
                       edge_idx: uint) {
            let edge_dir = edge_dir as uint;
            graph.edges[edge_idx].next_edge[edge_dir] =
                graph.nodes[*node_id].head_edge[edge_dir];
            graph.nodes[*node_id].head_edge[edge_dir] =
                edge_idx;
        }
    }

    fn expansion(graph: &Graph) {
        do self.iterate_until_fixed_point(~"Expansion", graph) |edge| {
            match edge.constraint {
              ConstrainRegSubVar(copy a_region, copy b_vid) => {
                let b_node = &graph.nodes[*b_vid];
                self.expand_node(a_region, b_vid, b_node)
              }
              ConstrainVarSubVar(copy a_vid, copy b_vid) => {
                match graph.nodes[*a_vid].value {
                  NoValue | ErrorValue => false,
                  Value(copy a_region) => {
                    let b_node = &graph.nodes[*b_vid];
                    self.expand_node(a_region, b_vid, b_node)
                  }
                }
              }
              ConstrainVarSubReg(*) => {
                // This is a contraction constraint.  Ignore it.
                false
              }
            }
        }
    }

    fn expand_node(a_region: region,
                   b_vid: RegionVid,
                   b_node: &GraphNode) -> bool {
        debug!("expand_node(%?, %? == %?)",
               a_region, b_vid, b_node.value);

        b_node.classification = Expanding;
        match b_node.value {
          NoValue => {
            debug!("Setting initial value of %? to %?", b_vid, a_region);

            b_node.value = Value(a_region);
            return true;
          }

          Value(copy cur_region) => {
            let lub = self.lub_concrete_regions(a_region, cur_region);
            if lub == cur_region {
                return false;
            }

            debug!("Expanding value of %? from %? to %?",
                   b_vid, cur_region, lub);

            b_node.value = Value(lub);
            return true;
          }

          ErrorValue => {
            return false;
          }
        }
    }

    fn contraction(graph: &Graph) {
        do self.iterate_until_fixed_point(~"Contraction", graph) |edge| {
            match edge.constraint {
              ConstrainRegSubVar(*) => {
                // This is an expansion constraint.  Ignore.
                false
              }
              ConstrainVarSubVar(copy a_vid, copy b_vid) => {
                match graph.nodes[*b_vid].value {
                  NoValue | ErrorValue => false,
                  Value(copy b_region) => {
                    let a_node = &graph.nodes[*a_vid];
                    self.contract_node(a_vid, a_node, b_region)
                  }
                }
              }
              ConstrainVarSubReg(copy a_vid, copy b_region) => {
                let a_node = &graph.nodes[*a_vid];
                self.contract_node(a_vid, a_node, b_region)
              }
            }
        }
    }

    fn contract_node(a_vid: RegionVid,
                     a_node: &GraphNode,
                     b_region: region) -> bool {
        debug!("contract_node(%? == %?/%?, %?)",
               a_vid, a_node.value, a_node.classification, b_region);

        return match a_node.value {
          NoValue => {
            assert a_node.classification == Contracting;
            a_node.value = Value(b_region);
            true // changed
          }

          ErrorValue => {
            false // no change
          }

          Value(copy a_region) => {
            match a_node.classification {
              Expanding => {
                check_node(&self, a_vid, a_node, a_region, b_region)
              }
              Contracting => {
                adjust_node(&self, a_vid, a_node, a_region, b_region)
              }
            }
          }
        };

        fn check_node(self: &RegionVarBindings,
                      a_vid: RegionVid,
                      a_node: &GraphNode,
                      a_region: region,
                      b_region: region) -> bool {
            if !self.is_subregion_of(a_region, b_region) {
                debug!("Setting %? to ErrorValue: %? not subregion of %?",
                       a_vid, a_region, b_region);
                a_node.value = ErrorValue;
            }
            false
        }

        fn adjust_node(self: &RegionVarBindings,
                       a_vid: RegionVid,
                       a_node: &GraphNode,
                       a_region: region,
                       b_region: region) -> bool {
            match self.glb_concrete_regions(a_region, b_region) {
              Ok(glb) => {
                if glb == a_region {
                    false
                } else {
                    debug!("Contracting value of %? from %? to %?",
                           a_vid, a_region, glb);
                    a_node.value = Value(glb);
                    true
                }
              }
              Err(_) => {
                a_node.value = ErrorValue;
                false
              }
            }
        }
    }

    fn iterate_until_fixed_point(
        tag: ~str,
        graph: &Graph,
        body: fn(edge: &GraphEdge) -> bool)
    {
        let mut iteration = 0;
        let mut changed = true;
        let num_edges = graph.edges.len();
        while changed {
            changed = false;
            iteration += 1;
            debug!("---- %s Iteration #%u", tag, iteration);
            for uint::range(0, num_edges) |edge_idx| {
                changed |= body(&graph.edges[edge_idx]);
                debug!(" >> Change after edge #%?: %?",
                       edge_idx, graph.edges[edge_idx]);
            }
        }
        debug!("---- %s Complete after %u iteration(s)", tag, iteration);
    }

    fn extract_regions_and_report_errors(graph: &Graph) -> ~[region] {
        let dup_map = TwoRegionsMap();
        graph.nodes.mapi(|idx, node| {
            match node.value {
              Value(v) => v,

              NoValue => {
                self.tcx.sess.span_err(
                    node.span,
                    fmt!("Unconstrained region variable #%u", idx));
                ty::re_static
              }

              ErrorValue => {
                let node_vid = RegionVid(idx);
                match node.classification {
                  Expanding => {
                    self.report_error_for_expanding_node(
                        graph, dup_map, node_vid);
                  }
                  Contracting => {
                    self.report_error_for_contracting_node(
                        graph, dup_map, node_vid);
                  }
                }
                ty::re_static
              }
            }
        })
    }

    // Used to suppress reporting the same basic error over and over
    fn is_reported(dup_map: TwoRegionsMap,
                   r_a: region,
                   r_b: region) -> bool {
        let key = TwoRegions { a: r_a, b: r_b };
        !dup_map.insert(key, ())
    }

    fn report_error_for_expanding_node(graph: &Graph,
                                       dup_map: TwoRegionsMap,
                                       node_idx: RegionVid) {
        // Errors in expanding nodes result from a lower-bound that is
        // not contained by an upper-bound.
        let lower_bounds =
            self.collect_concrete_regions(graph, node_idx, Incoming);
        let upper_bounds =
            self.collect_concrete_regions(graph, node_idx, Outgoing);

        for vec::each(lower_bounds) |lower_bound| {
            for vec::each(upper_bounds) |upper_bound| {
                if !self.is_subregion_of(lower_bound.region,
                                         upper_bound.region) {

                    if self.is_reported(dup_map,
                                        lower_bound.region,
                                        upper_bound.region) {
                        return;
                    }

                    self.tcx.sess.span_err(
                        self.var_spans[*node_idx],
                        fmt!("cannot infer an appropriate lifetime \
                              due to conflicting requirements"));

                    note_and_explain_region(
                        self.tcx,
                        ~"first, the lifetime cannot outlive ",
                        upper_bound.region,
                        ~"...");

                    self.tcx.sess.span_note(
                        upper_bound.span,
                        fmt!("...due to the following expression"));

                    note_and_explain_region(
                        self.tcx,
                        ~"but, the lifetime must be valid for ",
                        lower_bound.region,
                        ~"...");

                    self.tcx.sess.span_note(
                        lower_bound.span,
                        fmt!("...due to the following expression"));

                    return;
                }
            }
        }
    }

    fn report_error_for_contracting_node(graph: &Graph,
                                         dup_map: TwoRegionsMap,
                                         node_idx: RegionVid) {
        // Errors in contracting nodes result from two upper-bounds
        // that have no intersection.
        let upper_bounds = self.collect_concrete_regions(graph, node_idx,
                                                         Outgoing);

        for vec::each(upper_bounds) |upper_bound_1| {
            for vec::each(upper_bounds) |upper_bound_2| {
                match self.glb_concrete_regions(upper_bound_1.region,
                                                upper_bound_2.region) {
                  Ok(_) => {}
                  Err(_) => {

                    if self.is_reported(dup_map,
                                        upper_bound_1.region,
                                        upper_bound_2.region) {
                        return;
                    }

                    self.tcx.sess.span_err(
                        self.var_spans[*node_idx],
                        fmt!("cannot infer an appropriate lifetime \
                              due to conflicting requirements"));

                    note_and_explain_region(
                        self.tcx,
                        ~"first, the lifetime must be contained by ",
                        upper_bound_1.region,
                        ~"...");

                    self.tcx.sess.span_note(
                        upper_bound_1.span,
                        fmt!("...due to the following expression"));

                    note_and_explain_region(
                        self.tcx,
                        ~"but, the lifetime must also be contained by ",
                        upper_bound_2.region,
                        ~"...");

                    self.tcx.sess.span_note(
                        upper_bound_2.span,
                        fmt!("...due to the following expression"));

                    return;
                  }
                }
            }
        }
    }

    fn collect_concrete_regions(graph: &Graph,
                                orig_node_idx: RegionVid,
                                dir: Direction) -> ~[SpannedRegion] {
        let set = uint_hash();
        let mut stack = ~[orig_node_idx];
        set.insert(*orig_node_idx, ());
        let mut result = ~[];
        while !vec::is_empty(stack) {
            let node_idx = vec::pop(stack);
            for self.each_edge(graph, node_idx, dir) |edge| {
                match edge.constraint {
                  ConstrainVarSubVar(from_vid, to_vid) => {
                    let vid = match dir {
                      Incoming => from_vid,
                      Outgoing => to_vid
                    };
                    if set.insert(*vid, ()) {
                        vec::push(stack, vid);
                    }
                  }

                  ConstrainRegSubVar(region, _) => {
                    assert dir == Incoming;
                    vec::push(result, SpannedRegion {
                        region: region,
                        span: edge.span
                    });
                  }

                  ConstrainVarSubReg(_, region) => {
                    assert dir == Outgoing;
                    vec::push(result, SpannedRegion {
                        region: region,
                        span: edge.span
                    });
                  }
                }
            }
        }
        return result;
    }

    fn each_edge(graph: &Graph,
                 node_idx: RegionVid,
                 dir: Direction,
                 op: fn(edge: &GraphEdge) -> bool) {
        let mut edge_idx = graph.nodes[*node_idx].head_edge[dir as uint];
        while edge_idx != uint::max_value {
            let edge_ptr = &graph.edges[edge_idx];
            if !op(edge_ptr) {
                return;
            }
            edge_idx = edge_ptr.next_edge[dir as uint];
        }
    }
}
