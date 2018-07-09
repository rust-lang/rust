# Equality and associated types

This section covers how the trait system handles equality between
associated types. The full system consists of several moving parts,
which we will introduce one by one:

- Projection and the `Normalize` predicate
- Skolemization
- The `ProjectionEq` predicate
- Integration with unification

## Associated type projection and normalization

When a trait defines an associated type (e.g.,
[the `Item` type in the `IntoIterator` trait][intoiter-item]), that
type can be referenced by the user using an **associated type
projection** like `<Option<u32> as IntoIterator>::Item`. (Often,
though, people will use the shorthand syntax `T::Item` – presently,
that syntax is expanded during
["type collection"](./type-checking.html) into the explicit form,
though that is something we may want to change in the future.)

[intoiter-item]: https://doc.rust-lang.org/nightly/core/iter/trait.IntoIterator.html#associatedtype.Item

<a name="normalize"></a>

In some cases, associated type projections can be **normalized** –
that is, simplified – based on the types given in an impl. So, to
continue with our example, the impl of `IntoIterator` for `Option<T>`
declares (among other things) that `Item = T`:

```rust,ignore
impl<T> IntoIterator for Option<T> {
  type Item = T;
  ...
}
```

This means we can normalize the projection `<Option<u32> as
IntoIterator>::Item` to just `u32`.

In this case, the projection was a "monomorphic" one – that is, it
did not have any type parameters.  Monomorphic projections are special
because they can **always** be fully normalized – but often we can
normalize other associated type projections as well. For example,
`<Option<?T> as IntoIterator>::Item` (where `?T` is an inference
variable) can be normalized to just `?T`.

In our logic, normalization is defined by a predicate
`Normalize`. The `Normalize` clauses arise only from
impls. For example, the `impl` of `IntoIterator` for `Option<T>` that
we saw above would be lowered to a program clause like so:

```text
forall<T> {
    Normalize(<Option<T> as IntoIterator>::Item -> T) :-
        Implemented(Option<T>: IntoIterator)
}
```

where in this case, the one `Implemented` condition is always true.

(An aside: since we do not permit quantification over traits, this is
really more like a family of program clauses, one for each associated
type.)

We could apply that rule to normalize either of the examples that
we've seen so far.

## Skolemized associated types

Sometimes however we want to work with associated types that cannot be
normalized. For example, consider this function:

```rust,ignore
fn foo<T: IntoIterator>(...) { ... }
```

In this context, how would we normalize the type `T::Item`? Without
knowing what `T` is, we can't really do so. To represent this case, we
introduce a type called a **skolemized associated type
projection**. This is written like so `(IntoIterator::Item)<T>`. You
may note that it looks a lot like a regular type (e.g., `Option<T>`),
except that the "name" of the type is `(IntoIterator::Item)`. This is
not an accident: skolemized associated type projections work just like
ordinary types like `Vec<T>` when it comes to unification. That is,
they are only considered equal if (a) they are both references to the
same associated type, like `IntoIterator::Item` and (b) their type
arguments are equal.

Skolemized associated types are never written directly by the user.
They are used internally by the trait system only, as we will see
shortly.

## Projection equality

So far we have seen two ways to answer the question of "When can we
consider an associated type projection equal to another type?":

- the `Normalize` predicate could be used to transform associated type
  projections when we knew which impl was applicable;
- **skolemized** associated types can be used when we don't.

We now introduce the `ProjectionEq` predicate to bring those two cases
together. The `ProjectionEq` predicate looks like so:

```text
ProjectionEq(<T as IntoIterator>::Item = U)
```

and we will see that it can be proven *either* via normalization or
skolemization. As part of lowering an associated type declaration from
some trait, we create two program clauses for `ProjectionEq`:

```text
forall<T, U> {
    ProjectionEq(<T as IntoIterator>::Item = U) :-
        Normalize(<T as IntoIterator>::Item -> U)
}

forall<T> {
    ProjectionEq(<T as IntoIterator>::Item = (IntoIterator::Item)<T>)
}
```

These are the only two `ProjectionEq` program clauses we ever make for
any given associated item.

## Integration with unification

Now we are ready to discuss how associated type equality integrates
with unification. As described in the
[type inference](./type-inference.html) section, unification is
basically a procedure with a signature like this:

```text
Unify(A, B) = Result<(Subgoals, RegionConstraints), NoSolution>
```

In other words, we try to unify two things A and B. That procedure
might just fail, in which case we get back `Err(NoSolution)`. This
would happen, for example, if we tried to unify `u32` and `i32`.

The key point is that, on success, unification can also give back to
us a set of subgoals that still remain to be proven (it can also give
back region constraints, but those are not relevant here).

Whenever unification encounters an (unskolemized!) associated type
projection P being equated with some other type T, it always succeeds,
but it produces a subgoal `ProjectionEq(P = T)` that is propagated
back up. Thus it falls to the ordinary workings of the trait system
to process that constraint.

(If we unify two projections P1 and P2, then unification produces a
variable X and asks us to prove that `ProjectionEq(P1 = X)` and
`ProjectionEq(P2 = X)`. That used to be needed in an older system to
prevent cycles; I rather doubt it still is. -nmatsakis)
