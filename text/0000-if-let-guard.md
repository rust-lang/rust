- Feature Name: if_let_guard
- Start Date: 2018-01-15
- RFC PR: 
- Rust Issue: 

# Summary
[summary]: #summary

Allow `if let` guards in `match` expressions.

# Motivation
[motivation]: #motivation

This feature would greatly simplify some logic where we must match a pattern iff some value computed from the `match`-bound values has a certain form, where said value may be costly or impossible (due to affine semantics) to recompute in the match arm.

For further motivation, see the example in the guide-level explanation. Absent this feature, we might rather write the following:
```rust
match ui.wait_event() {
    KeyPress(mod_, key, datum) =>
        if let Some(action) = intercept(mod_, key) { act(action, datum) }
        else { accept!(KeyPress(mod_, key, datum)) /* can't re-use event verbatim if `datum` is non-`Copy` */ }
    ev => accept!(ev),
}
```

`accept` may in general be lengthy and inconvenient to move into another function, for example if it refers to many locals.

The code is clearer with an `if let` guard as follows.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

*(Adapted from Rust book)*

A *match guard* is an `if let` condition specified after the pattern in a `match` arm that also must match if the pattern matches in order for that arm to be chosen. Match guards are useful for expressing more complex ideas than a pattern alone allows.

The condition can use variables created in the pattern, and the match arm can use any variables bound in the `if let` pattern (as well as any bound in the `match` pattern, unless the `if let` expression moves out of them).

Let us consider an example which accepts a user-interface event (e.g. key press, pointer motion) and follows 1 of 2 paths: either we intercept it and take some action or deal with it normally (whatever that might mean here):
```rust
match ui.wait_event() {
    KeyPress(mod_, key, datum) if let Some(action) = intercept(mod_, key) => act(action, datum),
    ev => accept!(ev),
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This proposal would introduce syntax for a match arm: `pat if let guard_pat = guard_expr => body_expr` with semantics so the arm is chosen iff the argument of `match` matches `pat` and `guard_expr` matches `guard_pat`. The variables of `pat` are bound in `guard_expr`, and the variables of `pat` and `guard_pat` are bound in `body_expr`. The syntax is otherwise the same as for `if` guards. (Indeed, `if` guards become effectively syntactic sugar for `if let` guards.)

An arm may not have both an `if` and an `if let` guard.

# Drawbacks
[drawbacks]: #drawbacks

* It further complicates the grammar.
* It is ultimately syntactic sugar, but the transformation to present Rust is potentially non-obvious.

# Rationale and alternatives
[alternatives]: #alternatives

* The chief alternatives are to rewrite the guard as an `if` guard and a bind in the match arm, or in some cases into the argument of `match`; or to write the `if let` in the match arm and copy the rest of the `match` into the `else` branch — what can be done with this syntax can already be done in Rust (to the author's knowledge); this proposal is purely ergonomic, but in the author's opinion, the ergonomic win is significant.
* The proposed syntax feels natural by analogy to the `if` guard syntax we already have, as between `if` and `if let` expressions. No alternative syntaxes were considered.

# Unresolved questions
[unresolved]: #unresolved-questions

Questions in scope of this proposal: none yet known

Questions out of scope:

* Should we allow multiple guards? This proposal allows only a single `if let` guard. One can combine `if` guards with `&&` — [an RFC](https://github.com/rust-lang/rfcs/issues/929) to allow `&&` in `if let` already is, so we may want to follow that in future for `if let` guards also.
* What happens if `guard_expr` moves out of `pat` but fails to match? This is already a question for `if` guards and (to the author's knowledge) not formally specified anywhere — this proposal (implicitly) copies that behavior.
