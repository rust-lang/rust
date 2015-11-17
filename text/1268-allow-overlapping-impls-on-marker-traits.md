- Feature Name: Allow overlapping impls for marker traits
- Start Date: 2015-09-02
- RFC PR: https://github.com/rust-lang/rfcs/pull/1268
- Rust Issue: https://github.com/rust-lang/rust/issues/29864

# Summary

Preventing overlapping implementations of a trait makes complete sense in the
context of determining method dispatch. There must not be ambiguity in what code
will actually be run for a given type. However, for marker traits, there are no
associated methods for which to indicate ambiguity. There is no harm in a type
being marked as `Sync` for multiple reasons.

# Motivation

This is purely to improve the ergonomics of adding/implementing marker traits.
While specialization will certainly make all cases not covered today possible,
removing the restriction entirely will improve the ergonomics in several edge
cases.

Some examples include:

- the coercible trait design presents at [RFC #91][91];
- the `ExnSafe` trait proposed in [RFC #1236][1236].

[91]: https://github.com/rust-lang/rfcs/pull/91
[1236]: https://github.com/rust-lang/rfcs/pull/1236

# Detailed design

For the purpose of this RFC, the definition of a marker trait is a trait with no
associated items. The design here is quite straightforward. The following code
fails to compile today:

```rust
trait Marker<A> {}

struct GenericThing<A, B> {
    a: A,
    b: B,
}

impl<A, B> Marker<GenericThing<A, B>> for A {}
impl<A, B> Marker<GenericThing<A, B>> for B {}
```

The two impls are considered overlapping, as there is no way to prove currently
that `A` and `B` are not the same type. However, in the case of marker traits,
there is no actual reason that they couldn't be overlapping, as no code could
actually change based on the `impl`.

For a concrete use case, consider some setup like the following:

```rust
trait QuerySource {
    fn select<T, C: Selectable<T, Self>>(&self, columns: C) -> SelectSource<C, Self> {
        ...
    }
}

trait Column<T> {}
trait Table: QuerySource {}
trait Selectable<T, QS: QuerySource>: Column<T> {}

impl<T: Table, C: Column<T>> Selectable<T, T> for C {}
```

However, when the following becomes introduced:

```rust
struct JoinSource<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> QuerySource for JoinSource<Left, Right> where
    Left: Table + JoinTo<Right>,
    Right: Table,
{
    ...
}
```

It becomes impossible to satisfy the requirements of `select`. The following
impl is disallowed today:

```rust
impl<Left, Right, C> Selectable<Left, JoinSource<Left, Right>> for C where
    Left: Table + JoinTo<Right>,
    Right: Table,
    C: Column<Left>,
{}

impl<Left, Right, C> Selectable<Right, JoinSource<Left, Right>> for C where
    Left: Table + JoinTo<Right>,
    Right: Table,
    C: Column<Right>,
{}
```

Since `Left` and `Right` might be the same type, this causes an overlap.
However, there's also no reason to forbid the overlap. There is no way to work
around this today. Even if you write an impl that is more specific about the
tables, that would be considered a non-crate local blanket implementation. The
only way to write it today is to specify each column individually.

# Drawbacks

With this change, adding any methods to an existing marker trait, even
defaulted, would be a breaking change. Once specialization lands, this could
probably be considered an acceptable breakage.

# Alternatives

Once specialization lands, there does not appear to be a case that is impossible
to write, albeit with some additional boilerplate, as you'll have to manually
specify the empty impl for any overlap that might occur.

# Unresolved questions

**How can we implement this design?** Simply lifting the coherence
restrictions is easy enough, but we will encounter some challenges
when we come to test whether a given trait impl holds. For example, if
we have something like:

```rust
impl<T:Send> MarkerTrait for T { }
impl<T:Sync> MarkerTrait for T { }
```

means that a type `Foo: MarkerTrait` can hold *either* by `Foo: Send`
*or* by `Foo: Sync`. Today, we prefer to break down an obligation like
`Foo: MarkerTrait` into component obligations (e.g., `Foo: Send`). Due
to coherence, there is always one best way to do this (sort of ---
where clauses complicate matters). That is, except for complications
due to type inference, there is a best impl to choose. But under this
proposal, there would not be. Experimentation is needed (similar
concerns arise with the proposals around specialization, so it may be
that progress on that front will answer the questions raised here).

**Should we add some explicit way to indicate that this is a marker
trait?** This would address the drawback that adding items is a
backwards incompatible change.
