# Goals and clauses

In logic programming terms, a **goal** is something that you must
prove and a **clause** is something that you know is true. As
described in the [lowering to logic](./traits/lowering-to-logic.html)
chapter, Rust's trait solver is based on an extension of hereditary
harrop (HH) clauses, which extend traditional Prolog Horn clauses with
a few new superpowers.

## Goals and clauses meta structure

In Rust's solver, **goals** and **clauses** have the following forms
(note that the two definitions reference one another):

```text
Goal = DomainGoal           // defined in the section below
        | Goal && Goal
        | Goal || Goal
        | exists<K> { Goal }   // existential quantification
        | forall<K> { Goal }   // universal quantification
        | if (Clause) { Goal } // implication
        | true                 // something that's trivially true
        | ambiguous            // something that's never provable

Clause = DomainGoal
        | Clause :- Goal     // if can prove Goal, then Clause is true
        | Clause && Clause
        | forall<K> { Clause }

K = <type>     // a "kind"
    | <lifetime>
```

The proof procedure for these sorts of goals is actually quite
straightforward.  Essentially, it's a form of depth-first search. The
paper
["A Proof Procedure for the Logic of Hereditary Harrop Formulas"][pphhf]
gives the details.

[pphhf]: ./traits/bibliography.html#pphhf

<a name="domain-goals"></a>

## Domain goals

<a name="trait-ref"></a>

To define the set of *domain goals* in our system, we need to first
introduce a few simple formulations. A **trait reference** consists of
the name of a trait along with a suitable set of inputs P0..Pn:

```text
TraitRef = P0: TraitName<P1..Pn>
```

So, for example, `u32: Display` is a trait reference, as is `Vec<T>:
IntoIterator`. Note that Rust surface syntax also permits some extra
things, like associated type bindings (`Vec<T>: IntoIterator<Item =
T>`), that are not part of a trait reference.

<a name="projection"></a>

A **projection** consists of an associated item reference along with
its inputs P0..Pm:

```text
Projection = <P0 as TraitName<P1..Pn>>::AssocItem<Pn+1..Pm>
```

Given these, we can define a `DomainGoal` as follows:

```text
DomainGoal = Implemented(TraitRef)
            | ProjectionEq(Projection = Type)
            | Normalize(Projection -> Type)
            | FromEnv(TraitRef)
            | FromEnv(Projection = Type)
            | WellFormed(Type)
            | WellFormed(TraitRef)
            | WellFormed(Projection = Type)
            | Outlives(Type: Region)
            | Outlives(Region: Region)
```

Let's break down each one of these, one-by-one.

#### Implemented(TraitRef)
e.g. `Implemented(i32: Copy)`

True if the given trait is implemented for the given input types and lifetimes.

#### ProjectionEq(Projection = Type)
e.g. `ProjectionEq<T as Iterator>::Item = u8`

The given associated type `Projection` is equal to `Type`; this can be proved
with either normalization or using skolemized types. See [the section
on associated types](./traits/associated-types.html).

#### Normalize(Projection -> Type)
e.g. `ProjectionEq<T as Iterator>::Item -> u8`

The given associated type `Projection` can be [normalized][n] to `Type`.

As discussed in [the section on associated
types](./traits/associated-types.html), `Normalize` implies `ProjectionEq`,
but not vice versa. In general, proving `Normalize(<T as Trait>::Item -> U)`
also requires proving `Implemented(T: Trait)`.

[n]: ./traits/associated-types.html#normalize
[at]: ./traits/associated-types.html

#### FromEnv(TraitRef), FromEnv(Projection = Type)
e.g. `FromEnv(Self: Add<i32>)`

e.g. `FromEnv(<Self as StreamingIterator>::Item<'a> = &'a [u8])`

True if the inner `TraitRef` or projection equality is *assumed* to be true;
that is, if it can be derived from the in-scope where clauses.

For example, given the following function:

```rust
fn loud_clone<T: Clone>(stuff: &T) -> T {
    println!("cloning!");
    stuff.clone()
}
```

Inside the body of our function, we would have `FromEnv(T: Clone)`. In-scope
where clauses nest, so a function body inside an impl body inherits the
impl body's where clauses, too.

This and the next rule are used to implement [implied bounds]. As we'll see
in the section on lowering, `FromEnv(X)` implies `Implemented(X)`, but not
vice versa. This distinction is crucial to implied bounds.

#### WellFormed(Item)
These goals imply that the given item is *well-formed*.

We can talk about different types of items being well-formed:

**Types**, like `WellFormed(Vec<i32>)`, which is true in Rust, or
  `WellFormed(Vec<str>)`, which is not (because `str` is not `Sized`.)

**TraitRefs**, like `WellFormed(Vec<i32>: Clone)`.

**Projections**, like `WellFormed(T: Iterator<Item = u32>)`.

Well-formedness is important to [implied bounds]. In particular, the reason
it is okay to assume `FromEnv(T: Clone)` in the example above is that we
_also_ verify `WellFormed(T: Clone)` for each call site of `loud_clone`.

#### Outlives(Type: Region), Outlives(Region: Region)
e.g. `Outlives(&'a str: 'b)`, `Outlives('a: 'static)`

True if the given type or region on the left outlives the right-hand region.

<a name="coinductive"></a>

## Coinductive goals

Most goals in our system are "inductive". In an inductive goal,
circular reasoning is disallowed. Consider this example clause:

```text
    Implemented(Foo: Bar) :-
        Implemented(Foo: Bar).
```

Considered inductively, this clause is useless: if we are trying to
prove `Implemented(Foo: Bar)`, we would then recursively have to prove
`Implemented(Foo: Bar)`, and that cycle would continue ad infinitum
(the trait solver will terminate here, it would just consider that
`Implemented(Foo: Bar)` is not known to be true).

However, some goals are *co-inductive*. Simply put, this means that
cycles are OK. So, if `Bar` were a co-inductive trait, then the rule
above would be perfectly valid, and it would indicate that
`Implemented(Foo: Bar)` is true.

*Auto traits* are one example in Rust where co-inductive goals are used.
Consider the `Send` trait, and imagine that we have this struct:

```rust
struct Foo {
    next: Option<Box<Foo>>
}
```

The default rules for auto traits say that `Foo` is `Send` if the
types of its fields are `Send`. Therefore, we would have a rule like

```text
Implemented(Foo: Send) :-
    Implemented(Option<Box<Foo>>: Send).
```

As you can probably imagine, proving that `Option<Box<Foo>>: Send` is
going to wind up circularly requiring us to prove that `Foo: Send`
again. So this would be an example where we wind up in a cycle – but
that's ok, we *do* consider `Foo: Send` to hold, even though it
references itself.

In general, co-inductive traits are used in Rust trait solving when we
want to enumerate a fixed set of possibilities. In the case of auto
traits, we are enumerating the set of reachable types from a given
starting point (i.e., `Foo` can reach values of type
`Option<Box<Foo>>`, which implies it can reach values of type
`Box<Foo>`, and then of type `Foo`, and then the cycle is complete).

In addition to auto traits, `WellFormed` predicates are co-inductive.
These are used to achieve a similar "enumerate all the cases" pattern,
as described in the section on [implied bounds].

[implied bounds]: ./traits/lowering-rules.html#implied-bounds

## Incomplete chapter

Some topics yet to be written:

- Elaborate on the proof procedure
- SLG solving – introduce negative reasoning
