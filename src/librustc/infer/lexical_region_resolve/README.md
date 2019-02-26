# Region inference

> WARNING: This README is obsolete and will be removed soon! For
> more info on how the current borrowck works, see the [rustc guide].
>
> As of edition 2018, region inference is done using Non-lexical lifetimes,
> which is described in the guide and [this RFC].

[rustc guide]: https://rust-lang.github.io/rustc-guide/mir/borrowck.html
[this RFC]: https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md

## Terminology

Note that we use the terms region and lifetime interchangeably.

## Introduction

Region inference uses a somewhat more involved algorithm than type
inference. It is not the most efficient thing ever written though it
seems to work well enough in practice (famous last words).  The reason
that we use a different algorithm is because, unlike with types, it is
impractical to hand-annotate with regions (in some cases, there aren't
even the requisite syntactic forms).  So we have to get it right, and
it's worth spending more time on a more involved analysis.  Moreover,
regions are a simpler case than types: they don't have aggregate
structure, for example.

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

## The Region Hierarchy

### Without closures

Let's first consider the region hierarchy without thinking about
closures, because they add a lot of complications. The region
hierarchy *basically* mirrors the lexical structure of the code.
There is a region for every piece of 'evaluation' that occurs, meaning
every expression, block, and pattern (patterns are considered to
"execute" by testing the value they are applied to and creating any
relevant bindings).  So, for example:

```rust
fn foo(x: isize, y: isize) { // -+
//  +------------+           //  |
//  |      +-----+           //  |
//  |  +-+ +-+ +-+           //  |
//  |  | | | | | |           //  |
//  v  v v v v v v           //  |
    let z = x + y;           //  |
    ...                      //  |
}                            // -+

fn bar() { ... }
```

In this example, there is a region for the fn body block as a whole,
and then a subregion for the declaration of the local variable.
Within that, there are sublifetimes for the assignment pattern and
also the expression `x + y`. The expression itself has sublifetimes
for evaluating `x` and `y`.

#s## Function calls

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

```rust
self.foo(self.bar())
```

where both `foo` and `bar` are `&mut self` functions will always yield
an error.

Here is a more involved example (which is safe) so we can see what's
going on:

```rust
struct Foo { f: usize, g: usize }
// ...
fn add(p: &mut usize, v: usize) {
    *p += v;
}
// ...
fn inc(p: &mut usize) -> usize {
    *p += 1; *p
}
fn weird() {
    let mut x: Box<Foo> = box Foo { /* ... */ };
    'a: add(&mut (*x).f,
            'b: inc(&mut (*x).f)) // (..)
}
```

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

```rust
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
```

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

```rust
struct Foo { f: usize, g: usize }
// ...
fn add(p: &mut usize, v: usize) {
    *p += v;
}
// ...
fn consume(x: Box<Foo>) -> usize {
    x.f + x.g
}
fn weird() {
    let mut x: Box<Foo> = box Foo { ... };
    'a: add(&mut (*x).f, consume(x)) // (..)
}
```

In this case, the second argument to `add` actually consumes `x`, thus
invalidating the first argument.

So, for now, we exclude the `call` lifetimes from our model.
Eventually I would like to include them, but we will have to make the
borrow checker handle this situation correctly. In particular, if
there is a reference created whose lifetime does not enclose
the borrow expression, we must issue sufficient restrictions to ensure
that the pointee remains valid.

### Modeling closures

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
