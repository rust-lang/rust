- Start Date: 2014-08-28
- RFC PR #: [rust-lang/rfcs#199](https://github.com/rust-lang/rfcs/pull/199)
- Rust Issue #: [rust-lang/rust#16810](https://github.com/rust-lang/rust/issues/16810)

# Summary

This is a *conventions RFC* for settling naming conventions when there
are by value, by reference, and by mutable reference variants of an
operation.

# Motivation

Currently the libraries are not terribly consistent about how to
signal mut variants of functions; sometimes it is by a `mut_` prefix,
sometimes a `_mut` suffix, and occasionally with `_mut_` appearing in
the middle. These inconsistencies make APIs difficult to remember.

While there are arguments in favor of each of the positions, we stand
to gain a lot by standardizing, and to some degree we just need to
make a choice.

# Detailed design

Functions often come in multiple variants: immutably borrowed, mutably
borrowed, and owned.

The canonical example is iterator methods:

- `iter` works with immutably borrowed data
- `mut_iter` works with mutably borrowed data
- `move_iter` works with owned data

For iterators, the "default" (unmarked) variant is immutably borrowed.
In other cases, the default is owned.

The proposed rules depend on which variant is the default, but use
*suffixes* to mark variants in all cases.

## The rules

### Immutably borrowed by default

If `foo` uses/produces an immutable borrow by default, use:

* The `_mut` suffix (e.g. `foo_mut`) for the mutably borrowed variant.
* The `_move` suffix (e.g. `foo_move`) for the owned variant.

However, in the case of iterators, the moving variant can also be
understood as an `into` conversion, `into_iter`, and `for x in v.into_iter()`
reads arguably better than `for x in v.iter_move()`, so the convention is
`into_iter`.

**NOTE**: This convention covers only the *method* names for
  iterators, not the names of the iterator types. That will be the
  subject of a follow up RFC.

### Owned by default

If `foo` uses/produces owned data by default, use:

* The `_ref` suffix (e.g. `foo_ref`) for the immutably borrowed variant.
* The `_mut` suffix (e.g. `foo_mut`) for the mutably borrowed variant.

### Exceptions

For mutably borrowed variants, if the `mut` qualifier is part of a
type name (e.g. `as_mut_slice`), it should appear as it would appear
in the type.

### References to type names

Some places in the current libraries, we say things like `as_ref` and
`as_mut`, and others we say `get_ref` and `get_mut_ref`.

Proposal: generally standardize on `mut` as a shortening of `mut_ref`.


## The rationale

### Why suffixes?

Using a suffix makes it easier to visually group variants together,
especially when sorted alphabetically. It puts the emphasis on the
functionality, rather than the qualifier.

### Why `move`?

Historically, Rust has used `move` as a way to signal ownership
transfer and to connect to C++ terminology. The main disadvantage is
that it does not emphasize ownership, which is our current narrative.
On the other hand, in Rust all data is owned, so using `_owned` as a
qualifier is a bit strange.

The `Copy` trait poses a problem for any terminology about ownership
transfer. The proposed mental model is that with `Copy` data you are
"moving a copy".

See Alternatives for more discussion.

### Why `mut` rather then `mut_ref`?

It's shorter, and pairs like `as_ref` and `as_mut` have a pleasant harmony
that doesn't place emphasis on one kind of reference over the other.

# Alternatives

## Prefix or mixed qualifiers

Using prefixes for variants is another possibility, but there seems to
be little upside.

It's possible to rationalize our current mix of prefixes and suffixes
via
[grammatical distinctions](https://github.com/rust-lang/rust/issues/13660#issuecomment-43576378),
but this seems overly subtle and complex, and requires a strong
command of English grammar to work well.

## No suffix exception

The rules here make an exception when `mut` is part of a type name, as
in `as_mut_slice`, but we could instead *always* place the qualifier
as a suffix: `as_slice_mut`. This would make APIs more consistent in
some ways, less in others: conversion functions would no longer
consistently use a transcription of their type name.

This is perhaps not so bad, though, because as it is we often
abbreviate type names. In any case, we need a convention (separate
RFC) for how to refer to type names in methods.

## `owned` instead of `move`

The overall narrative about Rust has been evolving to focus on
*ownership* as the essential concept, with borrowing giving various
lesser forms of ownership, so `_owned` would be a reasonable
alternative to `_move`.

On the other hand, the `ref` variants do not say "borrowed", so in
some sense this choice is inconsistent. In addition, the terminology
is less familiar to those coming from C++.

## `val` instead of `owned`

Another option would be `val` or `value` instead of `owned`. This
suggestion plays into the "by reference" and "by value" distinction,
and so is even more congruent with `ref` than `move` is. On the other
hand, it's less clear/evocative than either `move` or `owned`.
