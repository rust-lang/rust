# Return Position Impl Trait In Trait

Return-position impl trait in trait (RPITIT) is conceptually (and as of
[#112988], literally) sugar that turns RPITs in trait methods into
generic associated types (GATs) without the user having to define that
GAT either on the trait side or impl side.

RPITIT was originally implemented in [#101224], which added support for
async fn in trait (AFIT), since the implementation for RPITIT came for
free as a part of implementing AFIT which had been RFC'd previously. It
was then RFC'd independently in [RFC 3425], which was recently approved
by T-lang.

## How does it work?

This doc is ordered mostly via the compilation pipeline:

1. AST lowering (AST -> HIR)
2. HIR ty lowering (HIR -> rustc_middle::ty data types)
3. typeck

### AST lowering

AST lowering for RPITITs is almost the same as lowering RPITs. We
still lower them as
[`hir::ItemKind::OpaqueTy`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.OpaqueTy.html).
The two differences are that:

We record `in_trait` for the opaque. This will signify that the opaque
is an RPITIT for HIR ty lowering, diagnostics that deal with HIR, etc.

We record `lifetime_mapping`s for the opaque type, described below.

#### Aside: Opaque lifetime duplication

*All opaques* (not just RPITITs) end up duplicating their captured
lifetimes into new lifetime parameters local to the opaque. The main
reason we do this is because RPITs need to be able to "reify"[^1] any
captured late-bound arguments, or make them into early-bound ones. This
is so they can be used as generic args for the opaque, and later to
instantiate hidden types. Since we don't know which lifetimes are early-
or late-bound during AST lowering, we just do this for all lifetimes.

[^1]: This is compiler-errors terminology, I'm not claiming it's accurate :^)

The main addition for RPITITs is that during lowering we track the
relationship between the captured lifetimes and the corresponding
duplicated lifetimes in an additional field,
[`OpaqueTy::lifetime_mapping`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.OpaqueTy.html#structfield.lifetime_mapping).
We use this lifetime mapping later on in `predicates_of` to install
bounds that enforce equality between these duplicated lifetimes and
their source lifetimes in order to properly typecheck these GATs, which
will be discussed below.

##### Note

It may be better if we were able to lower without duplicates and for
that I think we would need to stop distinguishing between early and late
bound lifetimes. So we would need a solution like [Account for
late-bound lifetimes in generics
#103448](https://github.com/rust-lang/rust/pull/103448) and then also a
PR similar to [Inherit function lifetimes for impl-trait
#103449](https://github.com/rust-lang/rust/pull/103449).

### HIR ty lowering

The main change to HIR ty lowering is that we lower `hir::TyKind::OpaqueDef`
for an RPITIT to a projection instead of an opaque, using a newly
synthesized def-id for a new associated type in the trait. We'll
describe how exactly we get this def-id in the next section.

This means that any time we call `lower_ty` on the RPITIT, we end up
getting a projection back instead of an opaque. This projection can then
be normalized to the right value -- either the original opaque if we're
in the trait, or the inferred type of the RPITIT if we're in an impl.

#### Lowering to synthetic associated types

Using query feeding, we synthesize new associated types on both the
trait side and impl side for RPITITs that show up in methods.

##### Lowering RPITITs in traits

When `tcx.associated_item_def_ids(trait_def_id)` is called on a trait to
gather all of the trait's associated types, the query previously just
returned the def-ids of the HIR items that are children of the trait.
After [#112988], additionally, for each method in the trait, we add the
def-ids returned by
`tcx.associated_types_for_impl_traits_in_associated_fn(trait_method_def_id)`,
which walks through each trait method, gathers any RPITITs that show up
in the signature, and then calls
`associated_type_for_impl_trait_in_trait` for each RPITIT, which
synthesizes a new associated type.

##### Lowering RPITITs in impls

Similarly, along with the impl's HIR items, for each impl method, we
additionally add all of the
`associated_types_for_impl_traits_in_associated_fn` for the impl method.
This calls `associated_type_for_impl_trait_in_impl`, which will
synthesize an associated type definition for each RPITIT that comes from
the corresponding trait method.

#### Synthesizing new associated types

We use query feeding
([`TyCtxtAt::create_def`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/query/plumbing/struct.TyCtxtAt.html#method.create_def))
to synthesize a new def-id for the synthetic GATs for each RPITIT.

Locally, most of rustc's queries match on the HIR of an item to compute
their values. Since the RPITIT doesn't really have HIR associated with
it, or at least not HIR that corresponds to an associated type, we must
compute many queries eagerly and
[feed](https://github.com/rust-lang/rust/pull/104940) them, like
`opt_def_kind`, `associated_item`, `visibility`, and`defaultness`.

The values for most of these queries is obvious, since the RPITIT
conceptually inherits most of its information from the parent function
(e.g. `visibility`), or because it's trivially knowable because it's an
associated type (`opt_def_kind`).

Some other queries are more involved, or cannot be fed, and we
document the interesting ones of those below:

##### `generics_of` for the trait

The GAT for an RPITIT conceptually inherits the same generics as the
RPIT it comes from. However, instead of having the method as the
generics' parent, the trait is the parent.

Currently we get away with taking the RPIT's generics and method
generics and flattening them both into a new generics list, preserving
the def-id of each of the parameters. (This may cause issues with
def-ids having the wrong parents, but in the worst case this will cause
diagnostics issues. If this ends up being an issue, we can synthesize
new def-ids for generic params whose parent is the GAT.)

<details>
<summary> <b>An illustrated example</b> </summary>

```rust
trait Foo {
    fn method<'early: 'early, 'late, T>() -> impl Sized + Captures<'early, 'late>;
}
```

Would desugar to...
```rust
trait Foo {
    //       vvvvvvvvv method's generics
    //                  vvvvvvvvvvvvvvvvvvvvvvvv opaque's generics
    type Gat<'early, T, 'early_duplicated, 'late>: Sized + Captures<'early_duplicated, 'late>;

    fn method<'early: 'early, 'late, T>() -> Self::Gat<'early, T, 'early, 'late>;
}
```
</details>

##### `generics_of` for the impl

The generics for an impl's GAT are a bit more interesting. They are
composed of RPITIT's own generics (from the trait definition), appended
onto the impl's methods generics. This has the same issue as above,
where the generics for the GAT have parameters whose def-ids have the
wrong parent, but this should only cause issues in diagnostics.

We could fix this similarly if we were to synthesize new generics
def-ids, but this can be done later in a forwards-compatible way,
perhaps by a interested new contributor.

##### `opt_rpitit_info`

Some queries rely on computing information that would result in cycles
if we were to feed them eagerly, like `explicit_predicates_of`.
Therefore we defer to the `predicates_of` provider to return the right
value for our RPITIT's GAT. We do this by detecting early on in the
query if the associated type is synthetic by using
[`opt_rpitit_info`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.opt_rpitit_info),
which returns `Some` if the associated type is synthetic.

Then, during a query like `explicit_predicates_of`, we can detect if an
associated type is synthetic like:

```rust
fn explicit_predicates_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ... {
    if let Some(rpitit_info) = tcx.opt_rpitit_info(def_id) {
        // Do something special for RPITITs...
        return ...;
    }

    // The regular computation which relies on access to the HIR of `def_id`.
}
```

##### `explicit_predicates_of`

RPITITs begin by copying the predicates of the method that defined it,
both on the trait and impl side.

Additionally, we install "bidirectional outlives" predicates.
Specifically, we add region-outlives predicates in both directions for
each captured early-bound lifetime that constrains it to be equal to the
duplicated early-bound lifetime that results from lowering. This is best
illustrated in an example:

```rust
trait Foo<'a> {
    fn bar() -> impl Sized + 'a;
}

// Desugars into...

trait Foo<'a> {
    type Gat<'a_duplicated>: Sized + 'a
    where
        'a: 'a_duplicated,
        'a_duplicated: 'a;
    //~^ Specifically, we should be able to assume that the
    // duplicated `'a_duplicated` lifetime always stays in
    // sync with the `'a` lifetime.

    fn bar() -> Self::Gat<'a>;
}
```

##### `assumed_wf_types`

The GATs in both the trait and impl inherit the `assumed_wf_types` of
the trait method that defines the RPITIT. This is to make sure that the
following code is well formed when lowered.

```rust
trait Foo {
    fn iter<'a, T>(x: &'a [T]) -> impl Iterator<Item = &'a T>;
}

// which is lowered to...

trait FooDesugared {
    type Iter<'a, T>: Iterator<Item = &'a T>;
    //~^ assumed wf: `&'a [T]`
    // Without assumed wf types, the GAT would not be well-formed on its own.

    fn iter<'a, T>(x: &'a [T]) -> Self::Iter<'a, T>;
}
```

Because `assumed_wf_types` is only defined for local def ids, in order
to properly implement `assumed_wf_types` for impls of foreign traits
with RPITs, we need to encode the assumed wf types of RPITITs in an
extern query
[`assumed_wf_types_for_rpitit`](https://github.com/rust-lang/rust/blob/a17c7968b727d8413801961fc4e89869b6ab00d3/compiler/rustc_ty_utils/src/implied_bounds.rs#L14).

### Typechecking

#### The RPITIT inference algorithm

The RPITIT inference algorithm is implemented in
[`collect_return_position_impl_trait_in_trait_tys`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.collect_return_position_impl_trait_in_trait_tys.html).

**High-level:** Given a impl method and a trait method, we take the
trait method and instantiate each RPITIT in the signature with an infer
var. We then equate this trait method signature with the impl method
signature, and process all obligations that fall out in order to infer
the type of all of the RPITITs in the method.

The method is also responsible for making sure that the hidden types for
each RPITIT actually satisfy the bounds of the `impl Trait`, i.e. that
if we infer `impl Trait = Foo`, that `Foo: Trait` holds.

<details>
    <summary><b>An example...</b></summary>

```rust
#![feature(return_position_impl_trait_in_trait)]

use std::ops::Deref;

trait Foo {
    fn bar() -> impl Deref<Target = impl Sized>;
             // ^- RPITIT ?0        ^- RPITIT ?1
}

impl Foo for () {
    fn bar() -> Box<String> { Box::new(String::new()) }
}
```

We end up with the trait signature that looks like `fn() -> ?0`, and
nested obligations `?0: Deref<Target = ?1>`, `?1: Sized`. The impl
signature is `fn() -> Box<String>`.

Equating these signatures gives us `?0 = Box<String>`, which then after
processing the obligation `Box<String>: Deref<Target = ?1>` gives us `?1
= String`, and the other obligation `String: Sized` evaluates to true.

By the end of the algorithm, we end up with a mapping between associated
type def-ids to concrete types inferred from the signature. We can then
use this mapping to implement `type_of` for the synthetic associated
types in the impl, since this mapping describes the type that should
come after the `=` in `type Assoc = ...` for each RPITIT.
</details>

##### Implied bounds in RPITIT hidden type inference

Since `collect_return_position_impl_trait_in_trait_tys` does fulfillment and
region resolution, we must provide it `assumed_wf_types` so that we can prove
region obligations with the same expected implied bounds as
`compare_method_predicate_entailment` does.

Since the return type of a method is understood to be one of the assumed WF
types, and we eagerly fold the return type with inference variables to do
opaque type inference, after opaque type inference, the return type will
resolve to contain the hidden types of the RPITITs. this would mean that the
hidden types of the RPITITs would be assumed to be well-formed without having
independently proven that they are. This resulted in a
[subtle unsoundness bug](https://github.com/rust-lang/rust/pull/116072). In
order to prevent this cyclic reasoning, we instead replace the hidden types of
the RPITITs in the return type of the method with *placeholders*, which lead
to no implied well-formedness bounds.

#### Default trait body

Type-checking a default trait body, like:

```rust
trait Foo {
    fn bar() -> impl Sized {
        1i32
    }
}
```

requires one interesting hack. We need to install a projection predicate
into the param-env of `Foo::bar` allowing us to assume that the RPITIT's
GAT normalizes to the RPITIT's opaque type. This relies on the
observation that a trait method and RPITIT's GAT will always be "in
sync". That is, one will only ever be overridden if the other one is as
well.

Compare this to a similar desugaring of the code above, which would fail
because we cannot rely on this same assumption:

```rust
#![feature(impl_trait_in_assoc_type)]
#![feature(associated_type_defaults)]

trait Foo {
    type RPITIT = impl Sized;

    fn bar() -> Self::RPITIT {
        01i32
    }
}
```

Failing because a down-stream impl could theoretically provide an
implementation for `RPITIT` without providing an implementation of
`bar`:

```text
error[E0308]: mismatched types
--> src/lib.rs:8:9
 |
5 |     type RPITIT = impl Sized;
 |     ------------------------- associated type defaults can't be assumed inside the trait defining them
6 |
7 |     fn bar() -> Self::RPITIT {
 |                 ------------ expected `<Self as Foo>::RPITIT` because of return type
8 |         01i32
 |         ^^^^^ expected associated type, found `i32`
 |
 = note: expected associated type `<Self as Foo>::RPITIT`
                       found type `i32`
```

#### Well-formedness checking

We check well-formedness of RPITITs just like regular associated types.

Since we added lifetime bounds in `predicates_of` that link the
duplicated early-bound lifetimes to their original lifetimes, and we
implemented `assumed_wf_types` which inherits the WF types of the method
from which the RPITIT originates ([#113704]), we have no issues
WF-checking the GAT as if it were a regular GAT.

### What's broken, what's weird, etc.

##### Specialization is super busted

The "default trait methods" described above does not interact well with
specialization, because we only install those projection bounds in trait
default methods, and not in impl methods. Given that specialization is
already pretty busted, I won't go into detail, but it's currently a bug
tracked in:
    * `tests/ui/impl-trait/in-trait/specialization-broken.rs`

##### Projections don't have variances

This code fails because projections don't have variances:
```rust
#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    // Note that the RPITIT below does *not* capture `'lt`.
    fn bar<'lt: 'lt>() -> impl Eq;
}

fn test<'a, 'b, T: Foo>() -> bool {
    <T as Foo>::bar::<'a>() == <T as Foo>::bar::<'b>()
    //~^ ERROR
    // (requires that `'a == 'b`)
}
```

This is because we can't relate `<T as Foo>::Rpitit<'a>` and `<T as
Foo>::Rpitit<'b>`, even if they don't capture their lifetime. If we were
using regular opaque types, this would work, because they would be
bivariant in that lifetime parameter:
```rust
#![feature(return_position_impl_trait_in_trait)]

fn bar<'lt: 'lt>() -> impl Eq {
    ()
}

fn test<'a, 'b>() -> bool {
    bar::<'a>() == bar::<'b>()
}
```

This is probably okay though, since RPITITs will likely have their
captures behavior changed to capture all in-scope lifetimes anyways.
This could also be relaxed later in a forwards-compatible way if we were
to consider variances of RPITITs when relating projections.

[#112988]: https://github.com/rust-lang/rust/pull/112988
[RFC 3425]: https://github.com/rust-lang/rfcs/pull/3425
[#101224]: https://github.com/rust-lang/rust/pull/101224
[#113704]: https://github.com/rust-lang/rust/pull/113704
