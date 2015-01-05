- Start Date: 2014-11-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/447
- Rust Issue: https://github.com/rust-lang/rust/issues/20598

# Summary

Disallow unconstrained type parameters from impls. In practice this
means that every type parameter must either:

1. appear in the trait reference of the impl, if any;
2. appear in the self type of the impl; or,
3. be bound as an associated type.

This is an informal description, see below for full details.

# Motivation

Today it is legal to have impls with type parameters that are
effectively unconstrainted. This RFC proses to make these illegal by
requiring that all impl type parameters must appear in either the self
type of the impl or, if the impl is a trait impl, an (input) type
parameter of the trait reference. Type parameters can also be constrained
by associated types.

There are many reasons to make this change. First, impls are not
explicitly instantiated or named, so there is no way for users to
manually specify the values of type variables; the values must be
inferred. If the type parameters do not appear in the trait reference
or self type, however, there is no basis on which to infer them; this
almost always yields an error in any case (unresolved type variable),
though there are some corner cases where the inferencer can find a
constraint.

Second, permitting unconstrained type parameters to appear on impls
can potentially lead to ill-defined semantics later on. The current
way that the language works for cross-crate inlining is that the body
of the method is effectively reproduced within the target crate, but
in a fully elaborated form where it is as if the user specified every
type explicitly that they possibly could. This should be sufficient to
reproduce the same trait selections, even if the crate adds additional
types and additional impls -- but this cannot be guaranteed if there
are free-floating type parameters on impls, since their values are not
written anywhere. (This semantics, incidentally, is not only
convenient, but also required if we wish to allow for specialization
as a possibility later on.)

Finally, there is little to no loss of expressiveness. The type
parameters in question can always be moved somewhere else.

Here are some examples to clarify what's allowed and disallowed. In
each case, we also clarify how the example can be rewritten to be
legal.

```rust
// Legal:
// - A is used in the self type.
// - B is used in the input trait type parameters.
impl<A,B> SomeTrait<Option<B>> for Foo<A> {
    type Output = Result<A, IoError>;
}

// Legal:
// - A and B are used in the self type
impl<A,B> Vec<(A,B)> {
    ...
}

// Illegal:
// - A does not appear in the self type nor trait type parameters.
//
// This sort of pattern can generally be written by making `Bar` carry
// `A` as a phantom type parameter, or by making `Elem` an input type
// of `Foo`.
impl<A> Foo for Bar {
    type Elem = A; // associated types do not count
    ...
}

// Illegal: B does not appear in the self type.
//
// Note that B could be moved to the method `get()` with no
// loss of expressiveness.
impl<A,B:Default> Foo<A> {
    fn do_something(&self) {
    }

    fn get(&self) -> B {
        B::Default
    }
}

// Legal: `U` does not appear in the input types,
// but it bound as an associated type of `T`.
impl<T,U> Foo for T
    where T : Bar<Out=U> {
}
```

# Detailed design

Type parameters are legal if they are "constrained" according to the
following inference rules:

```
If T appears in the impl trait reference,
  then: T is constrained

If T appears in the impl self type,
  then: T is constrained

If <T0 as Trait<T1...Tn>>::U == V appears in the impl predicates,
  and T0...Tn are constrained
  and T0 as Trait<T1...Tn> is not the impl trait reference
  then: V is constrained
```

The interesting rule is of course the final one. It says that type
parameters whose value is determined by an associated type reference
are legal. A simple example is:

```
impl<T,U> Foo for T
    where T : Bar<Out=U>
```

However, we have to be careful to avoid cases where the associated
type is an associated type of things that are not themselves
constrained:

```
impl<T,U,V> Foo for T
    where U: Bar<Out=V>
```

Similarly, the final clause in the rule aims to prevent an impl from
"self-referentially" constraining an output type parameter:

```
impl<T,U> Bar for T
    where T : Bar<Out=U>
```

This last case isn't that important because impls like this, when
used, tend to result in overflow in the compiler, but it's more
user-friendly to report an error earlier.

# Drawbacks

This pattern requires a non-local rewrite to reproduce:

```
impl<A> Foo for Bar {
    type Elem = A; // associated types do not count
    ...
}
```

# Alternatives

To make these type parameters well-defined, we could also create a
syntax for specifying impl type parameter instantiations and/or have
the compiler track the full tree of impl type parameter instantiations
at type-checking time and supply this to the translation phase. This
approach rules out the possibility of impl specialization.

# Unresolved questions

None.
