% Ownership variants [RFC #199]

> The guidelines below were approved by [RFC #199](https://github.com/rust-lang/rfcs/pull/199).

Functions often come in multiple variants: immutably borrowed, mutably
borrowed, and owned.

The right default depends on the function in question. Variants should
be marked through suffixes.

#### Immutably borrowed by default

If `foo` uses/produces an immutable borrow by default, use:

* The `_mut` suffix (e.g. `foo_mut`) for the mutably borrowed variant.
* The `_move` suffix (e.g. `foo_move`) for the owned variant.

#### Owned by default

If `foo` uses/produces owned data by default, use:

* The `_ref` suffix (e.g. `foo_ref`) for the immutably borrowed variant.
* The `_mut` suffix (e.g. `foo_mut`) for the mutably borrowed variant.

#### Exceptions

In the case of iterators, the moving variant can also be understood as
an `into` conversion, `into_iter`, and `for x in v.into_iter()` reads
arguably better than `for x in v.iter_move()`, so the convention is
`into_iter`.

For mutably borrowed variants, if the `mut` qualifier is part of a
type name (e.g. `as_mut_slice`), it should appear as it would appear
in the type.
