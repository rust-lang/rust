- Feature Name: place_left_arrow_syntax
- Start Date: 2015-07-28
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rather than trying to find a clever syntax for placement-new that leverages
the `in` keyword, instead use the syntax `PLACE_EXPR <- VALUE_EXPR`.

This takes advantage of the fact that `<-` was reserved as a token via
historical accident (that for once worked out in our favor).

# Motivation

One sentence: the syntax `a <- b` is short, can be parsed without
ambiguity, and is strongly connotated already with assignment.

Further text (essentially historical background):

There is much debate about what syntax to use for placement-new.
We started with `box (PLACE_EXPR) VALUE_EXPR`, then migrated towards
leveraging the `in` keyword instead of `box`, yielding `in (PLACE_EXPR) VALUE_EXPR`.

A lot of people disliked the `in (PLACE_EXPR) VALUE_EXPR` syntax
(see discussion from [RFC 809]).

[RFC 809]: https://github.com/rust-lang/rfcs/pull/809

In response to that discussion (and also due to personal preference)
I suggested the alternative syntax `in PLACE_EXPR { BLOCK_EXPR }`,
which is what landed when [RFC 809] was merged.

However, it is worth noting that this alternative syntax actually
failed to address a number of objections (some of which also
applied to the original `in (PLACE_EXPR) VALUE_EXPR` syntax):

 * [kennytm](https://github.com/rust-lang/rfcs/pull/809#issuecomment-73071324)

   > While in (place) value is syntactically unambiguous, it looks
   > completely unnatural as a statement alone, mainly because there
   > are no verbs in the correct place, and also using in alone is
   > usually associated with iteration (for x in y) and member
   > testing (elem in set).

 * [petrochenkov](https://github.com/rust-lang/rfcs/pull/809#issuecomment-73142168)

   > As C++11 experience has shown, when it's available, it will
   > become the default method of inserting elements in containers,
   > since it's never performing worse than "normal insertion" and
   > is often better. So it should really have as short and
   > convenient syntax as possible.

 * [p1start](https://github.com/rust-lang/rfcs/pull/809#issuecomment-73837430)

   > I’m not a fan of in <place> { <stmts> }, simply because the
   > requirement of a block suggests that it’s some kind of control
   > flow structure, or that all the statements inside will be
   > somehow run ‘in’ the given <place> (or perhaps, as @m13253
   > seems to have interpreted it, for all box expressions to go
   > into the given place). It would be our first syntactical
   > construct which is basically just an operator that has to
   > have a block operand.

I believe the `PLACE_EXPR <- VALUE_EXPR` syntax addresses all of the
above concerns.

# Detailed design

Extend the parser to parse `EXPR <- EXPR`.

`EXPR <- EXPR` is parsed into an AST form that is desugared in much
the same way that `in EXPR { BLOCK }` or `box (EXPR) EXPR` are
desugared (see [PR 27215]).

Thus the static and dynamic semantics of `PLACE_EXPR <- VALUE_EXPR`
are *equivalent* to `box (PLACE_EXPR) VALUE_EXPR`. Namely, it is
still an expression form that operates by:
 1. Evaluate the `PLACE_EXPR` to a place
 2. Evaluate `VALUE_EXPR` directly into the constructed place
 3. Return the finalized place value.

(See protocol as documented in [RFC 809] for more details here.)

[PR 27215]: https://github.com/rust-lang/rust/pull/27215

This parsing form can be separately feature-gated (this RFC was
written assuming that would be the procedure). However, since
placement-`in` landed very recently ([PR 27215]) and is still
feature-gated, we can also just fold this change in with
the pre-existing `placement_in_syntax` feature gate
(though that may be non-intuitive since the keyword `in` is
no longer part of the syntactic form).

This feature has already been prototyped, see [place-left-syntax branch].

[place-left-syntax branch]: https://github.com/rust-lang/rust/compare/rust-lang:master...pnkfelix:place-left-syntax

# Drawbacks

The only drawback I am aware of is this [comment from nikomataskis](https://github.com/rust-lang/rfcs/pull/809#issuecomment-73903777)

> the intent is less clear than with a devoted keyword.

Note however that this was stated with regards to a hypothetical
overloading of the `=` operator (at least that is my understanding).

I think the use of the `<-` operator can be considered sufficiently
"devoted" (i.e. separate) syntax to placate the above concern.

# Alternatives

See [different surface syntax] from the alternatives from [RFC 809].

[different surface syntax]: https://github.com/pnkfelix/rfcs/blob/fsk-placement-box-rfc/text/0000-placement-box.md#same-semantics-but-different-surface-syntax

# Unresolved questions

None

