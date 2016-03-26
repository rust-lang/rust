Region inference

# Terminology

Note that we use the terms region and lifetime interchangeably.

# Introduction

Region inference uses a somewhat more involved algorithm than type
inference. It is not the most efficient thing ever written though it
seems to work well enough in practice (famous last words).  The reason
that we use a different algorithm is because, unlike with types, it is
impractical to hand-annotate with regions (in some cases, there aren't
even the requisite syntactic forms).  So we have to get it right, and
it's worth spending more time on a more involved analysis.  Moreover,
regions are a simpler case than types: they don't have aggregate
structure, for example.

Unlike normal type inference, which is similar in spirit to H-M and thus
works progressively, the region type inference works by accumulating
constraints over the course of a function.  Finally, at the end of
processing a function, we process and solve the constraints all at
once.

The constraints are always of one of three possible forms:

- `ConstrainVarSubVar(Ri, Rj)` states that region variable Ri must be
  a subregion of Rj
- `ConstrainRegSubVar(R, Ri)` states that the concrete region R (which
  must not be a variable) must be a subregion of the variable Ri
- `ConstrainVarSubReg(Ri, R)` states the variable Ri shoudl be less
  than the concrete region R. This is kind of deprecated and ought to
  be replaced with a verify (they essentially play the same role).

In addition to constraints, we also gather up a set of "verifys"
(what, you don't think Verify is a noun? Get used to it my
friend!). These represent relations that must hold but which don't
influence inference proper. These take the form of:

- `VerifyRegSubReg(Ri, Rj)` indicates that Ri <= Rj must hold,
  where Rj is not an inference variable (and Ri may or may not contain
  one). This doesn't influence inference because we will already have
  inferred Ri to be as small as possible, so then we just test whether
  that result was less than Rj or not.
- `VerifyGenericBound(R, Vb)` is a more complex expression which tests
  that the region R must satisfy the bound `Vb`. The bounds themselves
  may have structure like "must outlive one of the following regions"
  or "must outlive ALL of the following regions. These bounds arise
  from constraints like `T: 'a` -- if we know that `T: 'b` and `T: 'c`
  (say, from where clauses), then we can conclude that `T: 'a` if `'b:
  'a` *or* `'c: 'a`.

# Building up the constraints

Variables and constraints are created using the following methods:

- `new_region_var()` creates a new, unconstrained region variable;
- `make_subregion(Ri, Rj)` states that Ri is a subregion of Rj
- `lub_regions(Ri, Rj) -> Rk` returns a region Rk which is
  the smallest region that is greater than both Ri and Rj
- `glb_regions(Ri, Rj) -> Rk` returns a region Rk which is
  the greatest region that is smaller than both Ri and Rj

The actual region resolution algorithm is not entirely
obvious, though it is also not overly complex.

## Snapshotting

It is also permitted to try (and rollback) changes to the graph.  This
is done by invoking `start_snapshot()`, which returns a value.  Then
later you can call `rollback_to()` which undoes the work.
Alternatively, you can call `commit()` which ends all snapshots.
Snapshots can be recursive---so you can start a snapshot when another
is in progress, but only the root snapshot can "commit".

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

## Computing the values for region variables

The algorithm is a simple dataflow algorithm. Each region variable
begins as empty. We iterate over the constraints, and for each constraint
we grow the relevant region variable to be as big as it must be to meet all the
constraints. This means the region variables can grow to be `'static` if
necessary.

## Verification

After all constraints are fully propoagated, we do a "verification"
step where we walk over the verify bounds and check that they are
satisfied. These bounds represent the "maximal" values that a region
variable can take on, basically.

# The Region Hierarchy

## Without closures

Let's first consider the region hierarchy without thinking about
closures, because they add a lot of complications. The region
hierarchy *basically* mirrors the lexical structure of the code.
There is a region for every piece of 'evaluation' that occurs, meaning
every expression, block, and pattern (patterns are considered to
"execute" by testing the value they are applied to and creating any
relevant bindings).  So, for example:

    fn foo(x: isize, y: isize) { // -+
    //  +------------+       //  |
    //  |      +-----+       //  |
    //  |  +-+ +-+ +-+       //  |
    //  |  | | | | | |       //  |
    //  v  v v v v v v       //  |
        let z = x + y;       //  |
        ...                  //  |
    }                        // -+

    fn bar() { ... }

In this example, there is a region for the fn body block as a whole,
and then a subregion for the declaration of the local variable.
Within that, there are sublifetimes for the assignment pattern and
also the expression `x + y`. The expression itself has sublifetimes
for evaluating `x` and `y`.

## Function calls

Function calls are a bit tricky. I will describe how we handle them
*now* and then a bit about how we can improve them (Issue #6268).

Consider a function call like `func(expr1, expr2)`, where `func`,
`arg1`, and `arg2` are all arbitrary expressions. Currently,
we construct a region hierarchy like:

    +----------------+
    |                |
    +--+ +---+  +---+|
    v  v v   v  v   vv
    func(expr1, expr2)

Here you can see that the call as a whole has a region and the
function plus arguments are subregions of that. As a side-effect of
this, we get a lot of spurious errors around nested calls, in
particular when combined with `&mut` functions. For example, a call
like this one

    self.foo(self.bar())

where both `foo` and `bar` are `&mut self` functions will always yield
an error.

Here is a more involved example (which is safe) so we can see what's
going on:

    struct Foo { f: usize, g: usize }
    ...
    fn add(p: &mut usize, v: usize) {
        *p += v;
    }
    ...
    fn inc(p: &mut usize) -> usize {
        *p += 1; *p
    }
    fn weird() {
        let mut x: Box<Foo> = box Foo { ... };
        'a: add(&mut (*x).f,
                'b: inc(&mut (*x).f)) // (..)
    }

The important part is the line marked `(..)` which contains a call to
`add()`. The first argument is a mutable borrow of the field `f`.  The
second argument also borrows the field `f`. Now, in the current borrow
checker, the first borrow is given the lifetime of the call to
`add()`, `'a`.  The second borrow is given the lifetime of `'b` of the
call to `inc()`. Because `'b` is considered to be a sublifetime of
`'a`, an error is reported since there are two co-existing mutable
borrows of the same data.

However, if we were to examine the lifetimes a bit more carefully, we
can see that this error is unnecessary. Let's examine the lifetimes
involved with `'a` in detail. We'll break apart all the steps involved
in a call expression:

    'a: {
        'a_arg1: let a_temp1: ... = add;
        'a_arg2: let a_temp2: &'a mut usize = &'a mut (*x).f;
        'a_arg3: let a_temp3: usize = {
            let b_temp1: ... = inc;
            let b_temp2: &'b = &'b mut (*x).f;
            'b_call: b_temp1(b_temp2)
        };
        'a_call: a_temp1(a_temp2, a_temp3) // (**)
    }

Here we see that the lifetime `'a` includes a number of substatements.
In particular, there is this lifetime I've called `'a_call` that
corresponds to the *actual execution of the function `add()`*, after
all arguments have been evaluated. There is a corresponding lifetime
`'b_call` for the execution of `inc()`. If we wanted to be precise
about it, the lifetime of the two borrows should be `'a_call` and
`'b_call` respectively, since the references that were created
will not be dereferenced except during the execution itself.

However, this model by itself is not sound. The reason is that
while the two references that are created will never be used
simultaneously, it is still true that the first reference is
*created* before the second argument is evaluated, and so even though
it will not be *dereferenced* during the evaluation of the second
argument, it can still be *invalidated* by that evaluation. Consider
this similar but unsound example:

    struct Foo { f: usize, g: usize }
    ...
    fn add(p: &mut usize, v: usize) {
        *p += v;
    }
    ...
    fn consume(x: Box<Foo>) -> usize {
        x.f + x.g
    }
    fn weird() {
        let mut x: Box<Foo> = box Foo { ... };
        'a: add(&mut (*x).f, consume(x)) // (..)
    }

In this case, the second argument to `add` actually consumes `x`, thus
invalidating the first argument.

So, for now, we exclude the `call` lifetimes from our model.
Eventually I would like to include them, but we will have to make the
borrow checker handle this situation correctly. In particular, if
there is a reference created whose lifetime does not enclose
the borrow expression, we must issue sufficient restrictions to ensure
that the pointee remains valid.

## Modeling closures

Integrating closures properly into the model is a bit of
work-in-progress. In an ideal world, we would model closures as
closely as possible after their desugared equivalents. That is, a
closure type would be modeled as a struct, and the region hierarchy of
different closure bodies would be completely distinct from all other
fns. We are generally moving in that direction but there are
complications in terms of the implementation.

In practice what we currently do is somewhat different. The basis for
the current approach is the observation that the only time that
regions from distinct fn bodies interact with one another is through
an upvar or the type of a fn parameter (since closures live in the fn
body namespace, they can in fact have fn parameters whose types
include regions from the surrounding fn body). For these cases, there
are separate mechanisms which ensure that the regions that appear in
upvars/parameters outlive the dynamic extent of each call to the
closure:

1. Types must outlive the region of any expression where they are used.
   For a closure type `C` to outlive a region `'r`, that implies that the
   types of all its upvars must outlive `'r`.
2. Parameters must outlive the region of any fn that they are passed to.

Therefore, we can -- sort of -- assume that any region from an
enclosing fns is larger than any region from one of its enclosed
fn. And that is precisely what we do: when building the region
hierarchy, each region lives in its own distinct subtree, but if we
are asked to compute the `LUB(r1, r2)` of two regions, and those
regions are in disjoint subtrees, we compare the lexical nesting of
the two regions.

*Ideas for improving the situation:* (FIXME #3696) The correctness
argument here is subtle and a bit hand-wavy. The ideal, as stated
earlier, would be to model things in such a way that it corresponds
more closely to the desugared code. The best approach for doing this
is a bit unclear: it may in fact be possible to *actually* desugar
before we start, but I don't think so. The main option that I've been
thinking through is imposing a "view shift" as we enter the fn body,
so that regions appearing in the types of fn parameters and upvars are
translated from being regions in the outer fn into free region
parameters, just as they would be if we applied the desugaring. The
challenge here is that type inference may not have fully run, so the
types may not be fully known: we could probably do this translation
lazilly, as type variables are instantiated. We would also have to
apply a kind of inverse translation to the return value. This would be
a good idea anyway, as right now it is possible for free regions
instantiated within the closure to leak into the parent: this
currently leads to type errors, since those regions cannot outlive any
expressions within the parent hierarchy. Much like the current
handling of closures, there are no known cases where this leads to a
type-checking accepting incorrect code (though it sometimes rejects
what might be considered correct code; see rust-lang/rust#22557), but
it still doesn't feel like the right approach.

### Skolemization

For a discussion on skolemization and higher-ranked subtyping, please
see the module `middle::infer::higher_ranked::doc`.
