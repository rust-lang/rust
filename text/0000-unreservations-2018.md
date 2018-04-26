- Feature Name: `unreservations`
- Start Date: 2018-04-26
- RFC PR: 
- Rust Issue: 

# Summary
[summary]: #summary

We unreserve:
+ `pure`
+ `unsize`
+ `sizeof`
+ `alignof`
+ `offsetof`
+ `priv`

# Motivation
[motivation]: #motivation

We are currently not using any of the reserved keywords listed in the [summary]
for anything in the language at the moment. We also have no intention of using
the keywords for anything in the future, and as such, we want to unreserve them
so that rustaceans can use them as identifiers.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

See the [reference-level-explanation].

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

[list of reserved keywords]: https://doc.rust-lang.org/book/second-edition/appendix-01-keywords.html#keywords-currently-in-use

The keywords listed below are removed from the
[list of reserved keywords] and are longer reserved such that they can be
used as general identifiers. This is done immediately and on edition 2015.

The keywords to unreserve are:
+ `pure`
+ `unsize`
+ `sizeof`
+ `alignof`
+ `offsetof`
+ `priv`

# Drawbacks
[drawbacks]: #drawbacks

The only drawback is that we're not able to use each listed word as a keyword
in the future, without a reservation in a new edition, if we realize that we
made a mistake.

See the rationale for potential risks with each keyword.

# Rationale and alternatives
[alternatives]: #alternatives

There's only one alternative: Not unreserving all listed / some keywords.

Not unreserving a keyword would make the word unavailable for use as an
identifier.

## General policy around unreservations

This RFC establishes a general rationale and policy for keyword unreservation:
*If we are not using a keyword for anything in the language, and we are sure
that we have no intention of using the keyword in the future, then it is
permissible to unreserve a keyword and it is motivated.
Additionally, if there is a desire for a keyword to be used as an identifier,
this can in some cases outweigh very hypothetical and speculative language features.*

## Rationale for `pure`

This keyword used to be used for `pure fn`, that is: as an effect.

[applicative]: http://hackage.haskell.org/package/base-4.11.1.0/docs/Control-Applicative.html#t:Applicative

When *generic associated types* (GATs) lands, it is likely that people would
like to use this in their [applicative functor][applicative] and monad libraries,
which speaks in favor of unreserving `pure`. This use case explicitly mentioned by [`@ubsan`](https://github.com/ubsan/) who requested that the keyword be unreserved for this purpose.

### Potential drawbacks

Examples / The reasons why we might want to keep `pure` reserved are:

#### 1. Effects

```rust
pure fn foo(x: Type) -> Type {
    ...
}
```

Here, `pure` denotes a deterministic function -- but we already have `const`
for more or less the same, and it is unlikely that we would introduce an effect
(or restriction thereof) that is essentially `const fn` but not entirely.
So this use case is unlikely to happen.

#### 2. Explicit *`Ok`-wrapping*

```rust
fn foo() -> Result<i32, Error> {
    if bar() {
        pure 0;
    }
    ...
}
```

desugars into:

```rust
fn foo() -> Result<i32, Error> {
    if bar() {
        return Try::from_ok(0);
    }
    ...
}
```

[Applicative laws]: https://en.wikibooks.org/wiki/Haskell/Applicative_functors#Applicative_functor_laws

While you might think that Haskell developers would be in favor of this,
that does not seem to be the case. Haskell developers over at
`#haskell @ freenode` were not particularly in favor of this use as `pure`
in this context as `pure` does not respect the [Applicative laws].
The desugaring is also not particularly obvious when `pure` is used.
If we did add sugar for explicit `Ok`-wrapping, we'd probably go with something
other than `pure`.

#### Summary

In both 1. and 2., `pure` can be contextual.
We also don't think that the drawbacks are significant for `pure`.

## Rationale for `unsize`

This would be a modifier on types, but we already have `<T: ?Sized>` and we
could have `T: !Sized` so there seems to be no need for keeping `unsized`.

## Rationale for `sizeof`, `alignof`, and `offsetof`

We already have [`std::mem::size_of`](https://doc.rust-lang.org/nightly/std/mem/fn.size_of.html) and similar which
are `const fn`s or can be.

A reason why we might want to keep these reserved is that they already exist in
the standard library, and so we might not want anyone to define these functions,
not because we will use them ourselves, but because it would be confusing,
and so the error messages could be improved saying
*"go look at `std::mem::size_of` instead"*. However, we believe it is better
to allow users the freedom to use these keywords instead.

## Rationale for `priv`

Here, `priv` is a privacy / visibility modifier on things like fields, and items.
An example:

```rust
priv struct Foo;
pub struct Bar {
    priv baz: u8
}
```

Since everything is already private by default, `priv` would only be an extra
hint that users can use to be more explict, but serves no other purpose.
Further, we could possibly use `pub(self)` for `priv` instead.

Permitting `priv` could also be confusing for readers. Consider for example:

```rust
pub struct Foo {
    priv bar: T,
    baz: U,
}
```

An unsuspecting reader can get the impression that `bar` is private but `baz`
is public. We could of course lint against this mixing, but it does not seem
worth the complexity.

For these reasons, the current proposal is to unreserve.

# Prior art
[prior-art]: #prior-art

Not applicable.

# Unresolved questions
[unresolved]: #unresolved-questions

There are none.
All reservations we will do should be resolved before merging the RFC.

# Appendix
[appendix]: #appendix

## Reserved keywords we probably don't want to unreserve

The following keywords are used in the nightly compiler and we are sure
that we want to keep them:

- `yield` - Generators
- `macro` - Macros 2.0

Additionally, there are known potential use cases / RFCs for:

- `become` - We might want this for guaranteed tail calls.
  See [the postponed RFC](https://github.com/rust-lang/rfcs/pull/1888).

- `typeof` - We might want this for hypothetical usages such as:
  ```rust
  fn foo(x: impl Bar, y: typeof(x)) { .. }
  ```

- `do` - We might want this for two uses:
  1. `do { .. } while cond;` loops.
  2. Haskell style do notation: `let az' = do { x <- ax; y <- ay(x); az };`.

- `abstract` - We might/would like this for:
  ```rust
  abstract type Foo: Copy + Debug + .. ;
  ```

- `override` - This could possibly used for:
  + OOP inheritance -- unlikely that we'll get such features.

  + specialization -- we do not annotate specialization on the overriding impl
  but rather say that the base impl is specializable with `default`,
  wherefore `override` does not make much sense.

  + delegation -- this usage was proposed in the delegations pre-RFC:

    ```rust
    impl TR for S {
        delegate * to f;

        #[override(from="f")]
        fn foo(&self) -> u32 {
            42
        }
    }
    ```

    which we could rewrite as:

    ```rust
    impl TR for S {
        delegate * to f;

        override(from f) fn foo(&self) -> u32 {
            42
        }
    }
    ```

## Possible future unreservations

### `box`

We use this in nightly for box patterns.
We might want to unreserve this eventually however.

### `virtual`

This annotation would be for something like virtual functions (see `dyn`).
However, we already have `dyn`, so why would we need `virtual`?
Assuming the following makes sense semantically (which we do not care about here),
we could easily write:

```rust
dyn fn foo(..) -> whatever { .. }
```

instead of:

```rust
virtual fn foo(..) -> whatever { .. }
```

However, there might be some use case related to specialization.
After specialization is stable, we would like to revisit unreservation of
`virtual`.

### `final`

The `final` keyword is currently reserved. It is used in Java to mean two
separate things:
1. "you can't extend (inheritance) this `class`",
2. "you can't mutate this variable",
    which we already have for `let` bindings by default.

A possible use for `final` for us might be for [`Frozen` ](https://internals.rust-lang.org/t/forever-immutable-owned-values/6807).
However, `Frozen` does not have many known uses other than for users who want
to be more strict about things. The word `final` might not be what Java users
would expect it to mean in this context, so it's probably not a good keyword
for `Frozen`.

However, there might be some use case related to specialization.
After specialization is stable, we would like to revisit unreservation of
`final`.
