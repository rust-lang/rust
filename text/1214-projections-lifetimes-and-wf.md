- Feature Name: N/A
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [rust-lang/rfcs#1214](https://github.com/rust-lang/rfcs/pull/1214)
- Rust Issue: [rust-lang/rust#27579](https://github.com/rust-lang/rust/issues/27579)

# Summary

Type system changes to address the outlives relation with respect to
projections, and to better enforce that all types are well-formed
(meaning that they respect their declared bounds). The current
implementation can be both unsound ([#24622]), inconvenient
([#23442]), and surprising ([#21748], [#25692]). The changes are as follows:

- Simplify the outlives relation to be syntactically based.
- Specify improved rules for the outlives relation and projections.
- Specify more specifically where WF bounds are enforced, covering
  several cases missing from the implementation.

The proposed changes here have been tested and found to cause only a
modest number of regressions (about two dozen root regressions were
previously found on crates.io; however, that run did not yet include
all the provisions from this RFC; updated numbers coming soon). In
order to minimize the impact on users, the plan is to first introduce
the changes in two stages:

1. Initially, warnings will be issued for cases that violate the rules
   specified in this RFC. These warnings are not lints and cannot be
   silenced except by correcting the code such that it type-checks
   under the new rules.
2. After one release cycle, those warnings will become errors.

Note that although the changes do cause regressions, they also cause
some code (like that in [#23442]) which currently gets errors to
compile successfully.

# Motivation

### TL;DR

This is a long detailed RFC that is attempting to specify in some
detail aspects of the type system that were underspecified or buggily
implemented before. This section just summarizes the effect on
existing Rust code in terms of changes that may be required.

**Warnings first, errors later.** Although the changes described in
this RFC are necessary for soundness (and many of them are straight-up
bugfixes), there is some impact on existing code. Therefore the plan
is to first issue warnings for a release cycle and then transition to
hard errors, so as to ease the migration.

**Associated type projections and lifetimes work more smoothly.** The
current rules for relating associated type projections (like `T::Foo`)
and lifetimes are somewhat cumbersome. The newer rules are more
flexible, so that e.g. we can deduce that `T::Foo: 'a` if `T: 'a`, and
similarly that `T::Foo` is well-formed if `T` is well-formed. As a
bonus, the new rules are also sound. ;)

**Simpler outlives relation.** The older definition for the outlives
relation `T: 'a` was rather subtle. The new rule basically says that
if all type/lifetime parameters appearing in the type `T` must outlive
`'a`, then `T: 'a` (though there can also be other ways for us to
decide that `T: 'a` is valid, such as in-scope where clauses). So for
example `fn(&'x X): 'a` if `'x: 'a` and `X: 'a` (presuming that `X` is
a type parameter). The older rules were based on what kind of data was
actually *reachable*, and hence accepted this type (since no data of
`&'x X` is reachable from a function pointer).  This change primarily
affects struct declarations, since they may now require additional
outlives bounds:

```rust
// OK now, but after this RFC requires `X: 'a`:
struct Foo<'a, X> {
    f: fn(&'a X) // (because of this field)
}
```

**More types are sanity checked.** Generally Rust requires that if you
have a type like `SomeStruct<T>`, then whatever where clauses are
declared on `SomeStruct` must hold for `T` (this is called being
"well-formed"). For example, if `SomeStruct` is declared like so:

```rust
struct SomeStruct<T:Eq> { .. }
```

then this implies that `SomeStruct<f32>` is ill-formed, since `f32`
does not implement `Eq` (just `PartialEq`). However, the current compiler
doesn't check this in associated type definitions:

```rust
impl Iterator for SomethingElse {
    type Item = SomeStruct<f32>; // accepted now, not after this RFC
}
```

Similarly, WF checking was skipped for trait object types and fn
arguments. This means that `fn(SomeStruct<f32>)` would be considered
well-formed today, though attempting to call the function would be an
error. Under this RFC, that fn type is not well-formed (though
sometimes when there are higher-ranked regions, WF checking may still
be deferred until the point where the fn is called).

There are a few other places where similar requirements were being
overlooked before but will now be enforced. For example, a number of
traits like the following were found in the wild:

```rust
trait Foo {
    // currently accepted, but should require that Self: Sized
    fn method(&self, value: Option<Self>);
}
```

To be well-formed, an `Option<T>` type requires that `T: Sized`.  In
this case, though `T=Self`, and `Self` is not `Sized` by
default. Therefore, this trait should be declared `trait Foo: Sized`
to be legal. The compiler is currently *attempting* to enforce these
rules, but many cases were overlooked in practice.

### Impact on crates.io

This RFC has been largely implemented and tested against crates.io. A
[total of 43 (root) crates are affected][crater-all] by the
changes. Interestingly, **the vast majority of warnings/errors that
occur are not due to new rules introduced by this RFC**, but rather
due to older rules being more correctly enforced.

Of the affected crates, **40 are receiving future compatibility
warnings and hence continue to build for the time being**. In the
[remaining three cases][crater-errors], it was not possible to isolate
the effects of the new rules, and hence the compiler reports an error
rather than a future compatibility warning.

What follows is a breakdown of the reason that crates on crates.io are
receiving errors or warnings. Each row in the table corresponds to one
of the explanations above.

Problem                       | Future-compat. warnings | Errors |
----------------------------- | ----------------------- | ------ |
More types are sanity checked |           35            |    3   |
Simpler outlives relation     |            5            |        |

As you can see, by far the largest source of problems is simply that
we are now sanity checking more types. This was always the intent, but
there were bugs in the compiler that led to it either skipping
checking altogether or only partially applying the rules. It is
interesting to drill down a bit further into the 38 warnings/errors
that resulted from more types being sanity checked in order to see
what kinds of mistakes are being caught:

Case | Problem                       | Number |
---- | ----------------------------- | ------ |
 1   | `Self: Sized` required        |   26   |
 2   | `Foo: Bar` required           |   11   |
 3   | Not object safe               |    1   |

An example of each case follows:

**Cases 1 and 2.** In the compiler today, types appearing in trait methods
are incompletely checked. This leads to a lot of traits with
insufficient bounds.  By far the most common example was that the
`Self` parameter would appear in a context where it must be sized,
usually when it is embedded within another type (e.g.,
`Option<Self>`). Here is an example:

```rust
trait Test {
    fn test(&self) -> Option<Self>;
    //                ~~~~~~~~~~~~
    //            Incorrectly permitted before.
}
```

Because `Option<T>` requires that `T: Sized`, this trait should be
declared as follows:

```rust
trait Test: Sized {
    fn test(&self) -> Option<Self>;
}
```

**Case 2.** Case 2 is the same as case 1, except that the missing
bound is some trait other than `Sized`, or in some cases an outlives
bound like `T: 'a`.

**Case 3.** The compiler currently permits non-object-safe traits to
be used as types, even if objects could never actually be created
([#21953]).

### Projections and the outlives relation

[RFC 192] introduced the outlives relation `T: 'a` and described the
rules that are used to decide when one type outlives a lifetime. In
particular, the RFC describes rules that govern how the compiler
determines what kind of borrowed data may be "hidden" by a generic
type. For example, given this function signature:

```rust
fn foo<'a,I>(x: &'a I)
    where I: Iterator
{ ... }
```

the compiler is able to use implied region bounds (described more
below) to automatically determine that:

- all borrowed content in the type `I` outlives the function body;
- all borrowed content in the type `I` outlives the lifetime `'a`.

When associated types were introduced in [RFC 195], some new rules
were required to decide when an "outlives relation" involving a
projection (e.g., `I::Item: 'a`) should hold. The initial rules were
[very conservative][#22246]. This led to the rules from [RFC 192]
being [adapted] to cover associated type projections like
`I::Item`. Unfortunately, these adapted rules are not ideal, and can
still lead to [annoying errors in some situations][#23442]. Finding a
better solution has been on the agenda for some time.

Simultaneously, we realized in [#24622] that the compiler had a bug
that caused it to erroneously assume that every projection like
`I::Item` outlived the current function body, just as it assumes that
type parameters like `I` outlive the current function body. **This bug
can lead to unsound behavior.** Unfortunately, simply implementing the
naive fix for #24622 exacerbates the shortcomings of the current rules
for projections, causing widespread compilation failures in all sorts
of reasonable and obviously correct code.

**This RFC describes modifications to the type system that both
restore soundness and make working with associated types more
convenient in some situations.** The changes are largely but not
completely backwards compatible.

### Well-formed types

A type is considered *well-formed* (WF) if it meets some simple
correctness criteria. For builtin types like `&'a T` or `[T]`, these
criteria are built into the language. For user-defined types like a
struct or an enum, the criteria are declared in the form of where
clauses. In general, all types that appear in the source and elsewhere
should be well-formed.

For example, consider this type, which combines a reference to a
hashmap and a vector of additional key/value pairs:

```rust
struct DeltaMap<'a, K, V> where K: Hash + 'a, V: 'a {
    base_map: &'a mut HashMap<K,V>,
    additional_values: Vec<(K,V)>
}
```

Here, the WF criteria for `DeltaMap<K,V>` are as follows:

- `K: Hash`, because of the where-clause,
- `K: 'a`, because of the where-clause,
- `V: 'a`, because of the where-clause
- `K: Sized`, because of the implicit `Sized` bound
- `V: Sized`, because of the implicit `Sized` bound

Let's look at those `K:'a` bounds a bit more closely. If you leave
them out, you will find that the the structure definition above does
not type-check. This is due to the requirement that the types of all
fields in a structure definition must be well-formed.  In this case,
the field `base_map` has the type `&'a mut HashMap<K,V>`, and this
type is only valid if `K: 'a` and `V: 'a` hold. Since we don't know
what `K` and `V` are, we have to surface this requirement in the form
of a where-clause, so that users of the struct know that they must
maintain this relationship in order for the struct to be interally
coherent.

#### An aside: explicit WF requirements on types

You might wonder why you have to write `K:Hash` and `K:'a` explicitly.
After all, they are obvious from the types of the fields. The reason
is that we want to make it possible to check whether a type like
`DeltaMap<'foo,T,U>` is well-formed *without* having to inspect the
types of the fields -- that is, in the current design, the only
information that we need to use to decide if `DeltaMap<'foo,T,U>` is
well-formed is the set of bounds and where-clauses.

This has real consequences on usability. It would be possible for the
compiler to infer bounds like `K:Hash` or `K:'a`, but the origin of
the bound might be quite remote. For example, we might have a series
of types like:

```rust
struct Wrap1<'a,K>(Wrap2<'a,K>);
struct Wrap2<'a,K>(Wrap3<'a,K>);
struct Wrap3<'a,K>(DeltaMap<'a,K,K>);
```

Now, for `Wrap1<'foo,T>` to be well-formed, `T:'foo` and `T:Hash` must
hold, but this is not obvious from the declaration of
`Wrap1`. Instead, you must trace deeply through its fields to find out
that this obligation exists.

#### Implied lifetime bounds

To help avoid undue annotation, Rust relies on implied lifetime bounds
in certain contexts. Currently, this is limited to fn bodies. The idea
is that for functions, we can make callers do some portion of the WF
validation, and let the callees just assume it has been done
already. (This is in contrast to the type definition, where we
required that the struct itself declares all of its requirements up
front in the form of where-clauses.)

To see this in action, consider a function that uses a `DeltaMap`:

```rust
fn foo<'a,K:Hash,V>(d: DeltaMap<'a,K,V>) { ... }
```

You'll notice that there are no `K:'a` or `V:'a` annotations required
here. This is due to *implied lifetime bounds*. Unlike structs, a
function's caller must examine not only the explicit bounds and
where-clauses, but *also* the argument and return types. When there
are generic type/lifetime parameters involved, the caller is in charge
of ensuring that those types are well-formed. (This is in contrast
with type definitions, where the type is in charge of figuring out its
own requirements and listing them in one place.)

As the name "implied lifetime bounds" suggests, we currently limit
implied bounds to region relationships. That is, we will implicitly
derive a bound like `K:'a` or `V:'a`, but not `K:Hash` -- this must
still be written manually. It might be a good idea to change this, but
that would be the topic of a separate RFC.

Currently, implied bound are limited to fn bodies. This RFC expands
the use of implied bounds to cover impl definitions as well, since
otherwise the annotation burden is quite painful. More on this in the
next section.

*NB.* There is an additional problem concerning the interaction of
implied bounds and contravariance ([#25860]). To better separate the
issues, this will be addressed in a follow-up RFC that should appear
shortly.

#### Missing WF checks

Unfortunately, the compiler currently fails to enforce WF in several
important cases. For example, the
[following program](http://is.gd/6JXjyg) is accepted:

```rust
struct MyType<T:Copy> { t: T }

trait ExampleTrait {
    type Output;
}

struct ExampleType;

impl ExampleTrait for ExampleType {
    type Output = MyType<Box<i32>>;
    //            ~~~~~~~~~~~~~~~~
    //                   |
    //   Note that `Box<i32>` is not `Copy`!
}
```

However, if we simply naively add the requirement that associated
types must be well-formed, this results in a large annotation burden
(see e.g. [PR 25701](https://github.com/rust-lang/rust/pull/25701/)).
For example, in practice, many iterator implementation break due to
region relationships:

```rust
impl<'a, T> IntoIterator for &'a LinkedList<T> {
   type Item = &'a T;
   ...
}
```

The problem here is that for `&'a T` to be well-formed, `T: 'a` must
hold, but that is not specified in the where clauses. This RFC
proposes using implied bounds to address this concern -- specifically,
every impl is permitted to assume that all types which appear in the
impl header (trait reference) are well-formed, and in turn each "user"
of an impl will validate this requirement whenever they project out of
a trait reference (e.g., to do a method call, or normalize an
associated type).

# Detailed design

This section dives into detail on the proposed type rules.

### A little type grammar

We extend the type grammar from [RFC 192] with projections and slice
types:

    T = scalar (i32, u32, ...)        // Boring stuff
      | X                             // Type variable
      | Id<P0..Pn>                    // Nominal type (struct, enum)
      | &r T                          // Reference (mut doesn't matter here)
      | O0..On+r                      // Object type
      | [T]                           // Slice type
      | for<r..> fn(T1..Tn) -> T0     // Function pointer
      | <P0 as Trait<P1..Pn>>::Id     // Projection
    P = r                             // Region name
      | T                             // Type
    O = for<r..> TraitId<P1..Pn>      // Object type fragment
    r = 'x                            // Region name

We'll use this to describe the rules in detail.

A quick note on terminology: an "object type fragment" is part of an
object type: so if you have `Box<FnMut()+Send>`, `FnMut()` and `Send`
are object type fragments. Object type fragments are identical to full
trait references, except that they do not have a self type (no `P0`).

### Syntactic definition of the outlives relation

The outlives relation is defined in purely syntactic terms as follows.
These are inference rules written in a primitive ASCII notation. :) As
part of defining the outlives relation, we need to track the set of
lifetimes that are bound within the type we are looking at.  Let's
call that set `R=<r0..rn>`. Initially, this set `R` is empty, but it
will grow as we traverse through types like fns or object fragments,
which can bind region names via `for<..>`.

#### Simple outlives rules

Here are the rules covering the simple cases, where no type parameters
or projections are involved:

    OutlivesScalar:
      --------------------------------------------------
      R ⊢ scalar: 'a

    OutlivesNominalType:
      ∀i. R ⊢ Pi: 'a
      --------------------------------------------------
      R ⊢ Id<P0..Pn>: 'a

    OutlivesReference:
      R ⊢ 'x: 'a
      R ⊢ T: 'a
      --------------------------------------------------
      R ⊢ &'x T: 'a

    OutlivesObject:
      ∀i. R ⊢ Oi: 'a
      R ⊢ 'x: 'a
      --------------------------------------------------
      R ⊢ O0..On+'x: 'a

    OutlivesFunction:
      ∀i. R,r.. ⊢ Ti: 'a
      --------------------------------------------------
      R ⊢ for<r..> fn(T1..Tn) -> T0: 'a

    OutlivesFragment:
      ∀i. R,r.. ⊢ Pi: 'a
      --------------------------------------------------
      R ⊢ for<r..> TraitId<P0..Pn>: 'a

#### Outlives for lifetimes

The outlives relation for lifetimes depends on whether the lifetime in
question was bound within a type or not. In the usual case, we decide
the relationship between two lifetimes by consulting the environment,
or using the reflexive property. Lifetimes representing scopes within
the current fn have a relationship derived from the code itself, while
lifetime parameters have relationships defined by where-clauses and
implied bounds.

    OutlivesRegionEnv:
      'x ∉ R               // not a bound region
      ('x: 'a) in Env      // derivable from where-clauses etc
      --------------------------------------------------
      R ⊢ 'x: 'a

    OutlivesRegionReflexive:
      --------------------------------------------------
      R ⊢ 'a: 'a

    OutlivesRegionTransitive:
      R ⊢ 'a: 'c
      R ⊢ 'c: 'b
      --------------------------------------------------
      R ⊢ 'a: 'b

For higher-ranked lifetimes, we simply ignore the relation, since the
lifetime is not yet known. This means for example that `for<'a> fn(&'a
i32): 'x` holds, even though we do not yet know what region `'a` is
(and in fact it may be instantiated many times with different values
on each call to the fn).

    OutlivesRegionBound:
      'x ∈ R               // bound region
      --------------------------------------------------
      R ⊢ 'x: 'a

#### Outlives for type parameters

For type parameters, the only way to draw "outlives" conclusions is to
find information in the environment (which is being threaded
implicitly here, since it is never modified). In terms of a Rust
program, this means both explicit where-clauses and implied bounds
derived from the signature (discussed below).

    OutlivesTypeParameterEnv:
      X: 'a in Env
      --------------------------------------------------
      R ⊢ X: 'a


#### Outlives for projections

Projections have the most possibilities. First, we may find
information in the in-scope where clauses, as with type parameters,
but we can also consult the trait definition to find bounds (consider
an associated type declared like `type Foo: 'static`). These rule only
apply if there are no higher-ranked lifetimes in the projection; for
simplicity's sake, we encode that by requiring an empty list of
higher-ranked lifetimes. (This is somewhat stricter than necessary,
but reflects the behavior of my prototype implementation.)

    OutlivesProjectionEnv:
      <P0 as Trait<P1..Pn>>::Id: 'b in Env
      <> ⊢ 'b: 'a
      --------------------------------------------------
      <> ⊢ <P0 as Trait<P1..Pn>>::Id: 'a

    OutlivesProjectionTraitDef:
      WC = [Xi => Pi] WhereClauses(Trait)
      <P0 as Trait<P1..Pn>>::Id: 'b in WC
      <> ⊢ 'b: 'a
      --------------------------------------------------
      <> ⊢ <P0 as Trait<P1..Pn>>::Id: 'a

All the rules covered so far already exist today. This last rule,
however, is not only new, it is the crucial insight of this RFC.  It
states that if all the components in a projection's trait reference
outlive `'a`, then the projection must outlive `'a`:

    OutlivesProjectionComponents:
      ∀i. R ⊢ Pi: 'a
      --------------------------------------------------
      R ⊢ <P0 as Trait<P1..Pn>>::Id: 'a

Given the importance of this rule, it's worth spending a bit of time
discussing it in more detail. The following explanation is fairly
informal. A more detailed look can be found in the appendix.

Let's begin with a concrete example of an iterator type, like
`std::vec::Iter<'a,T>`. We are interested in the projection of
`Iterator::Item`:

    <Iter<'a,T> as Iterator>::Item

or, in the more succint (but potentially ambiguous) form:

    Iter<'a,T>::Item

Since I'm going to be talking a lot about this type, let's just call
it `<PROJ>` for now. We would like to determine whether `<PROJ>: 'x` holds.

Now, the easy way to solve `<PROJ>: 'x` would be to normalize `<PROJ>`
by looking at the relevant impl:

```rust
impl<'b,U> Iterator for Iter<'b,U> {
    type Item = &'b U;
    ...
}
```

From this impl, we can conclude that `<PROJ> == &'a T`, and thus
reduce `<PROJ>: 'x` to `&'a T: 'x`, which in turn holds if `'a: 'x`
and `T: 'x` (from the rule `OutlivesReference`).

But often we are in a situation where we can't normalize the
projection (for example, a projection like `I::Item` where we only
know that `I: Iterator`). What can we do then? The rule
`OutlivesProjectionComponents` says that if we can conclude that every
lifetime/type parameter `Pi` to the trait reference outlives `'x`,
then we know that a projection from those parameters outlives `'x`. In
our example, the trait reference is `<Iter<'a,T> as Iterator>`, so
that means that if the type `Iter<'a,T>` outlives `'x`, then the
projection `<PROJ>` outlives `'x`. Now, you can see that this
trivially reduces to the same result as the normalization, since
`Iter<'a,T>: 'x` holds if `'a: 'x` and `T: 'x` (from the rule
`OutlivesNominalType`).

OK, so we've seen that applying the rule
`OutlivesProjectionComponents` comes up with the same result as
normalizing (at least in this case), and that's a good sign. But what
is the basis of the rule?

The basis of the rule comes from reasoning about the impl that we used
to do normalization. Let's consider that impl again, but this time
hide the actual type that was specified:

```rust
impl<'b,U> Iterator for Iter<'b,U> {
    type Item = /* <TYPE> */;
    ...
}
```

So when we normalize `<PROJ>`, we obtain the result by applying some
substitution `Θ` to `<TYPE>`. This substitution is a mapping from the
lifetime/type parameters on the impl to some specific values, such
that `<PROJ> == Θ <Iter<'b,U> as Iterator>::Item`. In this case, that
means `Θ` would be `['b => 'a, U => T]` (and of course `<TYPE>` would
be `&'b U`, but we're not supposed to rely on that).

The key idea for the `OutlivesProjectionComponents` is that the only
way that `<TYPE>` can *fail* to outlive `'x` is if either:

- it names some lifetime parameter `'p` where `'p: 'x` does not hold; or,
- it names some type parameter `X` where `X: 'x` does not hold.

Now, the only way that `<TYPE>` can refer to a parameter `P` is if it
is brought in by the substitution `Θ`. So, if we can just show that
all the types/lifetimes that in the range of `Θ` outlive `'x`, then we
know that `Θ <TYPE>` outlives `'x`.

Put yet another way: imagine that you have an impl with *no
parameters*, like:

```rust
impl Iterator for Foo {
    type Item = /* <TYPE> */;
}
```

Clearly, whatever `<TYPE>` is, it can only refer to the lifetime
`'static`.  So `<Foo as Iterator>::Item: 'static` holds. We know this
is true without ever knowing what `<TYPE>` is -- we just need to see
that the trait reference `<Foo as Iterator>` doesn't have any
lifetimes or type parameters in it, and hence the impl cannot refer to
any lifetime or type parameters.

#### Implementation complications

The current region inference code only permits constraints of the
form:

```
C = r0: r1
  | C AND C
```

This is convenient because a simple fixed-point iteration suffices to
find the minimal regions which satisfy the constraints.

Unfortunately, this constraint model does not scale to the outlives
rules for projections. Consider a trait reference like `<T as
Trait<'X>>::Item: 'Y`, where `'X` and `'Y` are both region variables
whose value is being inferred. At this point, there are several
inference rules which could potentially apply. Let us assume that
there is a where-clause in the environment like `<T as
Trait<'a>>::Item: 'b`. In that case, *if* `'X == 'a` and `'b: 'Y`,
then we could employ the `OutlivesProjectionEnv` rule. This would
correspond to a constraint set like:

```
C = 'X:'a AND 'a:'X AND 'b:'Y
```

Otherwise, if `T: 'a` and `'X: 'Y`, then we could use the
`OutlivesProjectionComponents` rule, which would require a constraint
set like:

```
C = C1 AND 'X:'Y
```

where `C1` is the constraint set for `T:'a`.

As you can see, these two rules yielded distinct constraint sets.
Ideally, we would combine them with an `OR` constraint, but no such
constraint is available. Adding such a constraint complicates how
inference works, since a fixed-point iteration is no longer
sufficient.

This complication is unfortunate, but to a large extent already exists
with where-clauses and trait matching (see e.g. [#21974]). (Moreover,
it seems to be inherent to the concept of assocated types, since they
take several inputs (the parameters to the trait) which may or may not
be related to the actual type definition in question.)

For the time being, the current implementation takes a pragmatic
approach based on heuristics. It first examines whether any region
bounds are declared in the trait and, if so, prefers to use
those. Otherwise, if there are region variables in the projection,
then it falls back to the `OutlivesProjectionComponents` rule. This is
always sufficient but may be stricter than necessary. If there are no
region variables in the projection, then it can simply run inference
to completion and check each of the other two rules in turn. (It is
still necessary to run inference because the bound may be a region
variable.) So far this approach has sufficed for all situations
encountered in practice. Eventually, we should extend the region
inferencer to a richer model that includes "OR" constraints.

### The WF relation

This section describes the "well-formed" relation. In
[previous RFCs][RFC 192], this was combined with the outlives
relation. We separate it here for reasons that shall become clear when
we discuss WF conditions on impls.

The WF relation is really pretty simple: it just says that a type is
"self-consistent". Typically, this would include validating scoping
(i.e., that you don't refer to a type parameter `X` if you didn't
declare one), but we'll take those basic conditions for granted.

    WfScalar:
      --------------------------------------------------
      R ⊢ scalar WF

    WfParameter:
      --------------------------------------------------
      R ⊢ X WF                  // where X is a type parameter

    WfTuple:
      ∀i. R ⊢ Ti WF
      ∀i<n. R ⊢ Ti: Sized       // the *last* field may be unsized
      --------------------------------------------------
      R ⊢ (T0..Tn) WF

    WfNominalType:
      ∀i. R ⊢ Pi Wf             // parameters must be WF,
      C = WhereClauses(Id)      // and the conditions declared on Id must hold...
      R ⊢ [P0..Pn] C            // ...after substituting parameters, of course
      --------------------------------------------------
      R ⊢ Id<P0..Pn> WF

    WfReference:
      R ⊢ T WF                  // T must be WF
      R ⊢ T: 'x                 // T must outlive 'x
      --------------------------------------------------
      R ⊢ &'x T WF

    WfSlice:
      R ⊢ T WF
      R ⊢ T: Sized
      --------------------------------------------------
      [T] WF

    WfProjection:
      ∀i. R ⊢ Pi WF             // all components well-formed
      R ⊢ <P0: Trait<P1..Pn>>   // the projection itself is valid
      --------------------------------------------------
      R ⊢ <P0 as Trait<P1..Pn>>::Id WF

#### WF checking and higher-ranked types

There are two places in Rust where types can introduce lifetime names
into scope: fns and trait objects. These have somewhat different rules
than the rest, simply because they modify the set `R` of bound
lifetime names. Let's start with the rule for fn types:

    WfFn:
      ∀i. R, r.. ⊢ Ti WF
      --------------------------------------------------
      R ⊢ for<r..> fn(T1..Tn) -> T0 WF

Basically, this rule adds the bound lifetimes to the set `R` and then
checks whether the argument and return type are well-formed. We'll see
in the next section that means that any requirements on those types
which reference bound identifiers are just assumed to hold, but the
remainder are checked. For example, if we have a type `HashSet<K>`
which requires that `K: Hash`, then `fn(HashSet<NoHash>)` would be
illegal since `NoHash: Hash` does not hold, but `for<'a>
fn(HashSet<&'a NoHash>)` *would* be legal, since `&'a NoHash: Hash`
involves a bound region `'a`. See the "Checking Conditions" section
for details.

Note that `fn` types do not require that `T0..Tn` be `Sized`.  This is
intentional. The limitation that only sized values can be passed as
argument (or returned) is enforced at the time when a fn is actually
called, as well as in actual fn definitions, but is not considered
fundamental to fn types thesmelves. There are several reasons for
this. For one thing, it's forwards compatible with passing DST by
value. For another, it means that non-defaulted trait methods to do
not have to show that their argument types are `Sized` (this will be
checked in the implementations, where more types are known). Since the
implicit `Self` type parameter is not `Sized` by default ([RFC 546]),
requiring that argument types be `Sized` in trait definitions proves
to be an annoying annotation burden.

The object type rule is similar, though it includes an extra clause:

    WfObject:
      rᵢ = union of implied region bounds from Oi
      ∀i. rᵢ: r
      ∀i. R ⊢ Oi WF
      --------------------------------------------------
      R ⊢ O0..On+r WF

The first two clauses here state that the explicit lifetime bound `r`
must be an approximation for the the implicit bounds `rᵢ` derived from
the trait definitions. That is, if you have a trait definition like

```rust
trait Foo: 'static { ... }
```

and a trait object like `Foo+'x`, when we require that `'static: 'x`
(which is true, clearly, but in some cases the implicit bounds from
traits are not `'static` but rather some named lifetime).

The next clause states that all object type fragments must be WF. An
object type fragment is WF if its components are WF:

    WfObjectFragment:
      ∀i. R, r.. ⊢ Pi
      TraitId is object safe
      --------------------------------------------------
      R ⊢ for<r..> TraitId<P1..Pn>

Note that we don't check the where clauses declared on the trait
itself. These are checked when the object is created. The reason not
to check them here is because the `Self` type is not known (this is an
object, after all), and hence we can't check them in general. (But see
*unresolved questions*.)

#### WF checking a trait reference

In some contexts, we want to check a trait reference, such as the ones
that appear in where clauses or type parameter bounds. The rules for
this are given here:

    WfTraitReference:
      ∀i. R, r.. ⊢ Pi
      C = WhereClauses(Id)      // and the conditions declared on Id must hold...
      R, r0...rn ⊢ [P0..Pn] C   // ...after substituting parameters, of course
      --------------------------------------------------
      R ⊢ for<r..> P0: TraitId<P1..Pn>

The rules are fairly straightforward. The components must be well formed,
and any where-clauses declared on the trait itself much hold.

#### Checking conditions

In various rules above, we have rules that declare that a where-clause
must hold, which have the form `R ̣⊢ WhereClause`. Here, `R` represents
the set of bound regions. It may well be that `WhereClause` does not
use any of the regions in `R`. In that case, we can ignore the
bound-regions and simple check that `WhereClause` holds. But if
`WhereClause` *does* refer to regions in `R`, then we simply consider
`R ⊢ WhereClause` to hold. Those conditions will be checked later when
the bound lifetimes are instantiated (either through a call or a
projection).

In practical terms, this means that if I have a type like:

```rust
struct Iterator<'a, T:'a> { ... }
```

and a function type like `for<'a> fn(i: Iterator<'a, T>)` then this
type is considered well-formed without having to show that `T: 'a`
holds. In terms of the rules, this is because we would wind up with a
constraint like `'a ⊢ T: 'a`.

However, if I have a type like

```rust
struct Foo<'a, T:Eq> { .. }
```

and a function type like `for<'a> fn(f: Foo<'a, T>)`, I still must
show that `T: Eq` holds for that function to be well-formed.  This is
because the condition which is geneated will be `'a ⊢ T: Eq`, but `'a`
is not referenced there.

#### Implied bounds

Implied bounds can be derived from the WF and outlives relations.  The
implied bounds from a type `T` are given by expanding the requirements
that `T: WF`. Since we currently limit ourselves to implied region
bounds, we we are interesting in extracting requirements of the form:

- `'a:'r`, where two regions must be related;
- `X:'r`, where a type parameter `X` outlives a region; or,
- `<T as Trait<..>>::Id: 'r`, where a projection outlives a region.

Some caution is required around projections when deriving implied
bounds. If we encounter a requirement that e.g. `X::Id: 'r`, we cannot
for example deduce that `X: 'r` must hold. This is because while `X:
'r` is *sufficient* for `X::Id: 'r` to hold, it is not *necessary* for
`X::Id: 'r` to hold. So we can only conclude that `X::Id: 'r` holds,
and not `X: 'r`.

#### When should we check the WF relation and under what conditions?

Currently the compiler performs WF checking in a somewhat haphazard
way: in some cases (such as impls), it omits checking WF, but in
others (such as fn bodies), it checks WF when it should not have
to. Partly that is due to the fact that the compiler currently
connects the WF and outlives relationship into one thing, rather than
separating them as described here.

**Constants/statics.** The type of a constant or static can be checked
for WF in an empty environment.

**Struct/enum declarations.** In a struct/enum declaration, we should
check that all field types are WF, given the bounds and where-clauses
from the struct declaration. Also check that where-clauses are well-formed.

**Function items.** For function items, the environment consists of
all the where-clauses from the fn, as well as implied bounds derived
from the fn's argument types. These are then used to check that the
following are well-formed:

- argument types;
- return type;
- where clauses;
- types of local variables.

These WF requirements are imposed at each fn or associated fn
definition (as well as within trait items).

**Trait impls.** In a trait impl, we assume that all types appearing
in the impl header are well-formed. This means that the initial
environment for an impl consists of the impl where-clauses and implied
bounds derived from its header. Example: Given an impl like
`impl<'a,T> SomeTrait for &'a T`, the environment would be `T: Sized`
(explicit where-clause) and `T: 'a` (implied bound derived from `&'a
T`). This environment is used as the starting point for checking the
items:

- Where-clauses declared on the trait must be WF.
- Associated types must be WF in the trait environment.
- The types of associated constants must be WF in the trait environment.
- Associated fns are checked just like regular function items, but
  with the additional implied bounds from the impl signature.

**Inherent impls.** In an inherent impl, we can assume that the self
type is well-formed, but otherwise check the methods as if they were
normal functions. We must check that all items are well-formed, along with
the where clauses declared on the impl.

**Trait declarations.** Trait declarations (and defaults) are checked
in the same fashion as impls, except that there are no implied bounds
from the impl header. We must check that all items are well-formed,
along with the where clauses declared on the trait.

**Type aliases.** Type aliases are currently not checked for WF, since
they are considered transparent to type-checking. It's not clear that
this is the best policy, but it seems harmless, since the WF rules
will still be applied to the expanded version. See the *Unresolved
Questions* for some discussion on the alternatives here.

Several points in the list above made use of *implied bounds* based on
assuming that various types were WF. We have to ensure that those
bounds are checked on the reciprocal side, as follows:

**Fns being called.** Before calling a fn, we check that its argument
and return types are WF. This check takes place after all
higher-ranked lifetimes have been instantiated. Checking the argument
types ensures that the implied bounds due to argument types are
correct. Checking the return type ensures that the resulting type of
the call is WF.

**Method calls, "UFCS" notation for fns and constants.** These are the
two ways to project a value out of a trait reference. A method call or
UFCS resolution will require that the trait reference is WF according
to the rules given above.

**Normalizing associated type references.** Whenever a projection type
like `T::Foo` is normalized, we will require that the trait reference
is WF.

# Drawbacks

N/A

# Alternatives

I'm not aware of any appealing alternatives.

# Unresolved questions

**Best policy for type aliases.** The current policy is not to check
type aliases, since they are transparent to type-checking, and hence
their expansion can be checked instead. This is coherent, though
somewhat confusing in terms of the interaction with projections, since
we frequently cannot resolve projections without at least minimal
bounds (i.e., `type IteratorAndItem<T:Iterator> = (T::Item,
T)`). Still, full-checking of WF on type aliases seems to just mean
more annotation with little benefit. It might be nice to keep the
current policy and later, if/when we adopt a more full notion of
implied bounds, rationalize it by saying that the suitable bounds for
a type alias are implied by its expansion.

**For trait object type fragments, should we check WF conditions when
we can?** For example, if you have:

```rust
trait HashSet<K:Hash>
```

should an object like `Box<HashSet<NotHash>>` be illegal? It seems
like that would be inline with our "best effort" approach to bound
regions, so probably yes.

[RFC 192]: https://github.com/rust-lang/rfcs/blob/master/text/0192-bounds-on-object-and-generic-types.md
[RFC 195]: https://github.com/rust-lang/rfcs/blob/master/text/0195-associated-items.md
[RFC 447]: https://github.com/rust-lang/rfcs/blob/master/text/0447-no-unused-impl-parameters.md
[#21748]: https://github.com/rust-lang/rust/issues/21748
[#23442]: https://github.com/rust-lang/rust/issues/23442
[#24622]: https://github.com/rust-lang/rust/issues/24622
[#22436]: https://github.com/rust-lang/rust/pull/22436
[#22246]: https://github.com/rust-lang/rust/issues/22246
[#25860]: https://github.com/rust-lang/rust/issues/25860
[#25692]: https://github.com/rust-lang/rust/issues/25692
[adapted]: https://github.com/rust-lang/rust/issues/22246#issuecomment-74186523
[#22077]: https://github.com/rust-lang/rust/issues/22077
[#24461]: https://github.com/rust-lang/rust/pull/24461
[#21974]: https://github.com/rust-lang/rust/issues/21974
[RFC 546]: 0546-Self-not-sized-by-default.md

# Appendix

The informal explanation glossed over some details. This appendix
tries to be a bit more thorough with how it is that we can conclude
that a projection outlives `'a` if its inputs outlive `'a`. To start,
let's specify the projection `<PROJ>` as:

    <P0 as Trait<P1...Pn>>::Id

where `P` can be a lifetime or type parameter as appropriate.

Then we know that there exists some impl of the form:

```rust
impl<X0..Xn> Trait<Q1..Qn> for Q0 {
    type Id = T;
}
```

Here again, `X` can be a lifetime or type parameter name, and `Q` can
be any lifetime or type parameter.

Let `Θ` be a suitable substitution `[Xi => Ri]` such that `∀i. Θ Qi ==
Pi` (in other words, so that the impl applies to the projection). Then
the normalized form of `<PROJ>` is `Θ T`. Note that because trait
matching is invariant, the types must be exactly equal.

[RFC 447] and [#24461] require that a parameter `Xi` can only appear
in `T` if it is *constrained* by the trait reference `<Q0 as
Trait<Q1..Qn>>`. The full definition of *constrained* appears below,
but informally it means roughly that `Xi` appears in `Q0..Qn`
somewhere outside of a projection. Let's call the constrained set of
parameters `Constrained(Q0..Qn)`.

Recall the rule `OutlivesProjectionComponents`:

    OutlivesProjectionComponents:
      ∀i. R ⊢ Pi: 'a
      --------------------------------------------------
      R ⊢ <P0 as Trait<P1..Pn>>::Id: 'a

We aim to show that `∀i. R ⊢ Pi: 'a` implies `R ⊢ (Θ T): 'a`, which implies
that this rule is a sound approximation for normalization.  The
argument follows from two lemmas ("proofs" for these lemmas are
sketched below):

1. First, we show that if `R ⊢ Pi: 'a`, then every "subcomponent" `P'`
   of `Pi` outlives `'a`.  The idea here is that each variable `Xi`
   from the impl will match against and extract some subcomponent `P'`
   of `Pi`, and we wish to show that the subcomponent `P'` extracted
   by `Xi` outlives `'a`.
2. Then we will show that the type `θ T` outlives `'a` if, for each of
   the in-scope parameters `Xi`, `Θ Xi: 'a`.

**Definition 1.** `Constrained(T)` defines the set of type/lifetime
parameters that are *constrained* by a type. This set is found just by
recursing over and extracting all subcomponents *except* for those
found in a projection. This is because a type like `X::Foo` does not
constrain what type `X` can take on, rather it uses `X` as an input to
compute a result:

    Constrained(scalar) = {}
    Constrained(X) = {X}
    Constrained(&'x T) = {'x} | Constrained(T)
    Constrained(O0..On+'x) = Union(Constrained(Oi)) | {'x}
    Constrained([T]) = Constrained(T),
    Constrained(for<..> fn(T1..Tn) -> T0) = Union(Constrained(Ti))
    Constrained(<P0 as Trait<P1..Pn>>::Id) = {} // empty set

**Definition 2.** `Constrained('a) = {'a}`. In other words, a lifetime
reference just constraints itself.

**Lemma 1:** Given `R ⊢ P: 'a`, `P = [X => P'] Q`, and `X ∈ Constrained(Q)`,
then `R ⊢ P': 'a`. Proceed by induction and by cases over the form of `P`:

1. If `P` is a scalar or parameter, there are no subcomponents, so `P'=P`.
2. For nominal types, references, objects, and function types, either
   `P'=P` or `P'` is some subcomponent of `P`. The appropriate "outlives"
   rules all require that all subcomponents outlive `'a`, and hence
   the conclusion follows by induction.
3. If `P'` is a projection, that implies that `P'=P`.
   - Otherwise, `Q` must be a projection, and in that case, `Constrained(Q)` would be
     the empty set.

**Lemma 2:** Given that `FV(T) ∈ X`, `∀i. Ri: 'a`, then `[X => R] T:
'a`. In other words, if all the type/lifetime parameters that appear
in a type outlive `'a`, then the type outlives `'a`. Follows by
inspection of the outlives rules.

# Edit History

[RFC1592] - amend to require that tuple fields be sized

[crater-errors]: https://gist.github.com/nikomatsakis/2f851e2accfa7ba2830d#root-regressions-sorted-by-rank
[crater-all]: https://gist.github.com/nikomatsakis/364fae49de18268680f2#root-regressions-sorted-by-rank
[#21953]: https://github.com/rust-lang/rust/issues/21953
[RFC1592]: https://github.com/rust-lang/rfcs/pull/1592