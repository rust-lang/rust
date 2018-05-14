- Feature Name: `unreserve_proc`
- Start Date: 2018-04-26
- RFC PR: [rust-lang/rfcs#2420](https://github.com/rust-lang/rfcs/pull/2420)
- Rust Issue: N/A. Already implemented.

# Summary
[summary]: #summary

The keyword `proc` gets unreserved.

# Motivation
[motivation]: #motivation

We are currently not using `proc` as a keyword for anything in the language.
Currently, `proc` is a reserved keyword for future use. However, we have
no intention of using the keyword for anything in the future, and as such,
we want to unreserve it so that rustaceans can use it as identifiers.

In the specific case of `proc`, it is a useful identifier for many things.
In particular, it is useful when dealing with processes, OS internals and
kernel development.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

See the [reference-level-explanation].

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

[list of reserved keywords]: https://doc.rust-lang.org/book/second-edition/appendix-01-keywords.html#keywords-currently-in-use

The keyword `proc` is removed from the [list of reserved keywords] and is no
longer reserved. This is done immediately and on edition 2015.

# Drawbacks
[drawbacks]: #drawbacks

The only drawback is that we're not able to use `proc` as a keyword in the
future, without a reservation in a new edition, if we realize that we made
a mistake.

[arrow]: https://downloads.haskell.org/~ghc/7.8.1/docs/html/users_guide/arrow-notation.html

The keyword `proc` could be used for some [`Arrow` notation][arrow] as used in
Haskell. However, `proc` notation is rarely used in Haskell since `Arrow`s are
not generally understood; and if something is not well understood by one of the
most academically inclined of communities of users, it is doubly a bad fit for
Rust which has a community mixed with users used to FP, systemsy and dynamically
checked programming languages. Moreover, `Arrow`s would most likely require HKTs
which we might not get.

# Rationale and alternatives
[alternatives]: #alternatives

There's only one alternative: Not doing anything.

Previously, this used to be the keyword used for `move |..| { .. }` closures,
but `proc` is no longer used for anything.

Not unreserving this keyword would make the word unavailable for use as an
identifier.

# Prior art
[prior-art]: #prior-art

Not applicable.

# Unresolved questions
[unresolved]: #unresolved-questions

There are none.
