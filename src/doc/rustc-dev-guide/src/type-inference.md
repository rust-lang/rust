# Type inference

<!-- toc -->

Type inference is the process of automatic detection of the type of an
expression.

It is what allows Rust to work with fewer or no type annotations,
making things easier for users:

```rust
fn main() {
    let mut things = vec![];
    things.push("thing");
}
```

Here, the type of `things` is *inferred* to be `Vec<&str>` because of the value
we push into `things`.

The type inference is based on the standard Hindley-Milner (HM) type inference
algorithm, but extended in various ways to accommodate subtyping, region
inference, and higher-ranked types.

## A note on terminology

We use the notation `?T` to refer to inference variables, also called
existential variables.

We use the terms "region" and "lifetime" interchangeably. Both refer to
the `'a` in `&'a T`.

The term "bound region" refers to a region that is bound in a function
signature, such as the `'a` in `for<'a> fn(&'a u32)`. A region is
"free" if it is not bound.

## Creating an inference context

You create an inference context by doing something like
the following:

```rust,ignore
let infcx = tcx.infer_ctxt().build();
// Use the inference context `infcx` here.
```

`infcx` has the type `InferCtxt<'tcx>`, the same `'tcx` lifetime as on
the `tcx` it was built from.

The `tcx.infer_ctxt` method actually returns a builder, which means
there are some kinds of configuration you can do before the `infcx` is
created. See `InferCtxtBuilder` for more information.

<a id="vars"></a>

## Inference variables

The main purpose of the inference context is to house a bunch of
**inference variables** – these represent types or regions whose precise
value is not yet known, but will be uncovered as we perform type-checking.

If you're familiar with the basic ideas of unification from H-M type
systems, or logic languages like Prolog, this is the same concept. If
you're not, you might want to read a tutorial on how H-M type
inference works, or perhaps this blog post on
[unification in the Chalk project].

[Unification in the Chalk project]: http://smallcultfollowing.com/babysteps/blog/2017/03/25/unification-in-chalk-part-1/

All told, the inference context stores five kinds of inference variables
(as of <!-- date-check --> March 2023):

- Type variables, which come in three varieties:
  - General type variables (the most common). These can be unified with any
    type.
  - Integral type variables, which can only be unified with an integral type,
    and arise from an integer literal expression like `22`.
  - Float type variables, which can only be unified with a float type, and
    arise from a float literal expression like `22.0`.
- Region variables, which represent lifetimes, and arise all over the place.
- Const variables, which represent constants.

All the type variables work in much the same way: you can create a new
type variable, and what you get is `Ty<'tcx>` representing an
unresolved type `?T`. Then later you can apply the various operations
that the inferencer supports, such as equality or subtyping, and it
will possibly **instantiate** (or **bind**) that `?T` to a specific
value as a result.

The region variables work somewhat differently, and are described
below in a separate section.

## Enforcing equality / subtyping

The most basic operations you can perform in the type inferencer is
**equality**, which forces two types `T` and `U` to be the same. The
recommended way to add an equality constraint is to use the `at`
method, roughly like so:

```rust,ignore
infcx.at(...).eq(t, u);
```

The first `at()` call provides a bit of context, i.e. why you are
doing this unification, and in what environment, and the `eq` method
performs the actual equality constraint.

When you equate things, you force them to be precisely equal. Equating
returns an `InferResult` – if it returns `Err(err)`, then equating
failed, and the enclosing `TypeError` will tell you what went wrong.

The success case is perhaps more interesting. The "primary" return
type of `eq` is `()` – that is, when it succeeds, it doesn't return a
value of any particular interest. Rather, it is executed for its
side-effects of constraining type variables and so forth. However, the
actual return type is not `()`, but rather `InferOk<()>`. The
`InferOk` type is used to carry extra trait obligations – your job is
to ensure that these are fulfilled (typically by enrolling them in a
fulfillment context). See the [trait chapter] for more background on that.

[trait chapter]: traits/resolution.html

You can similarly enforce subtyping through `infcx.at(..).sub(..)`. The same
basic concepts as above apply.

## "Trying" equality

Sometimes you would like to know if it is *possible* to equate two
types without error.  You can test that with `infcx.can_eq` (or
`infcx.can_sub` for subtyping). If this returns `Ok`, then equality
is possible – but in all cases, any side-effects are reversed.

Be aware, though, that the success or failure of these methods is always
**modulo regions**. That is, two types `&'a u32` and `&'b u32` will
return `Ok` for `can_eq`, even if `'a != 'b`.  This falls out from the
"two-phase" nature of how we solve region constraints.

## Snapshots

As described in the previous section on `can_eq`, often it is useful
to be able to do a series of operations and then roll back their
side-effects. This is done for various reasons: one of them is to be
able to backtrack, trying out multiple possibilities before settling
on which path to take. Another is in order to ensure that a series of
smaller changes take place atomically or not at all.

To allow for this, the inference context supports a `snapshot` method.
When you call it, it will start recording changes that occur from the
operations you perform. When you are done, you can either invoke
`rollback_to`, which will undo those changes, or else `confirm`, which
will make them permanent. Snapshots can be nested as long as you follow
a stack-like discipline.

Rather than use snapshots directly, it is often helpful to use the
methods like `commit_if_ok` or `probe` that encapsulate higher-level
patterns.

## Subtyping obligations

One thing worth discussing is subtyping obligations. When you force
two types to be a subtype, like `?T <: i32`, we can often convert those
into equality constraints. This follows from Rust's rather limited notion
of subtyping: so, in the above case, `?T <: i32` is equivalent to `?T = i32`.

However, in some cases we have to be more careful. For example, when
regions are involved. So if you have `?T <: &'a i32`, what we would do
is to first "generalize" `&'a i32` into a type with a region variable:
`&'?b i32`, and then unify `?T` with that (`?T = &'?b i32`). We then
relate this new variable with the original bound:

```text
&'?b i32 <: &'a i32
```

This will result in a region constraint (see below) of `'?b: 'a`.

One final interesting case is relating two unbound type variables,
like `?T <: ?U`.  In that case, we can't make progress, so we enqueue
an obligation `Subtype(?T, ?U)` and return it via the `InferOk`
mechanism. You'll have to try again when more details about `?T` or
`?U` are known.

## Region constraints

Regions are inferenced somewhat differently from types. Rather than
eagerly unifying things, we simply collect constraints as we go, but
make (almost) no attempt to solve regions. These constraints have the
form of an "outlives" constraint:

```text
'a: 'b
```

Actually the code tends to view them as a subregion relation, but it's the same
idea:

```text
'b <= 'a
```

(There are various other kinds of constraints, such as "verifys"; see
the [`region_constraints`] module for details.)

There is one case where we do some amount of eager unification. If you have an
equality constraint between two regions

```text
'a = 'b
```

we will record that fact in a unification table. You can then use
[`opportunistic_resolve_var`] to convert `'b` to `'a` (or vice
versa). This is sometimes needed to ensure termination of fixed-point
algorithms.

[`region_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/region_constraints/index.html
[`opportunistic_resolve_var`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/region_constraints/struct.RegionConstraintCollector.html#method.opportunistic_resolve_var

## Solving region constraints

Region constraints are only solved at the very end of
typechecking, once all other constraints are known and
all other obligations have been proven. There are two
ways to solve region constraints right now: lexical and
non-lexical. Eventually there will only be one.

An exception here is the leak-check which is used during trait solving
and relies on region constraints containing higher-ranked regions. Region
constraints in the root universe (i.e. not arising from a `for<'a>`) must
not influence the trait system, as these regions are all erased during
codegen.

To solve **lexical** region constraints, you invoke
[`resolve_regions_and_report_errors`].  This "closes" the region
constraint process and invokes the [`lexical_region_resolve`] code. Once
this is done, any further attempt to equate or create a subtyping
relationship will yield an ICE.

The NLL solver (actually, the MIR type-checker) does things slightly
differently. It uses canonical queries for trait solving which use
[`take_and_reset_region_constraints`] at the end. This extracts all of the
outlives constraints added during the canonical query. This is required
as the NLL solver must not only know *what* regions outlive each other,
but also *where*. Finally, the NLL solver invokes [`take_region_var_origins`],
providing all region variables to the solver.

[`resolve_regions_and_report_errors`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.resolve_regions_and_report_errors
[`lexical_region_resolve`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/lexical_region_resolve/index.html
[`take_and_reset_region_constraints`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.take_and_reset_region_constraints
[`take_region_var_origins`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.take_region_var_origins

## Lexical region resolution

Lexical region resolution is done by initially assigning each region
variable to an empty value. We then process each outlives constraint
repeatedly, growing region variables until a fixed-point is reached.
Region variables can be grown using a least-upper-bound relation on
the region lattice in a fairly straightforward fashion.
