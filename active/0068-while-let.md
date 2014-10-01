- Start Date: 2014-08-27
- RFC PR: https://github.com/rust-lang/rfcs/pull/214
- Rust Issue: https://github.com/rust-lang/rust/issues/17687

# Summary

Introduce a new `while let PAT = EXPR { BODY }` construct. This allows for using a refutable pattern
match (with optional variable binding) as the condition of a loop.

# Motivation

Just as `if let` was inspired by Swift, it turns out Swift supports `while let` as well. This was
not discovered until much too late to include it in the `if let` RFC. It turns out that this sort of
looping is actually useful on occasion. For example, the desugaring `for` loop is actually a variant
on this; if `while let` existed it could have been implemented to map `for PAT in EXPR { BODY }` to

```rust
// the match here is so `for` can accept an rvalue for the iterator,
// and was used in the "real" desugaring version.
match &mut EXPR {
    i => {
        while let Some(PAT) = i.next() {
            BODY
        }
    }
}
```

(note that the non-desugared form of `for` is no longer equivalent).

More generally, this construct can be used any time looping + pattern-matching is desired.

This also makes the language a bit more consistent; right now, any condition that can be used with
`if` can be used with `while`. The new `if let` adds a form of `if` that doesn't map to `while`.
Supporting `while let` restores the equivalence of these two control-flow constructs.

# Detailed design

`while let` operates similarly to `if let`, in that it desugars to existing syntax. Specifically,
the syntax

```rust
['ident:] while let PAT = EXPR {
    BODY
}
```

desugars to

```rust
['ident:] loop {
    match EXPR {
        PAT => BODY,
        _ => break
    }
}
```

Just as with `if let`, an irrefutable pattern given to `while let` is considered an error. This is
largely an artifact of the fact that the desugared `match` ends up with an unreachable pattern,
and is not actually a goal of this syntax. The error may be suppressed in the future, which would be
a backwards-compatible change.

Just as with `if let`, `while let` will be introduced under a feature gate (named `while_let`).

# Drawbacks

Yet another addition to the grammar. Unlike `if let`, it's not obvious how useful this syntax will
be.

# Alternatives

As with `if let`, this could plausibly be done with a macro, but it would be ugly and produce bad
error messages.

`while let` could be extended to support alternative patterns, just as match arms do. This is not
part of the main proposal for the same reason it was left out of `if let`, which is that a) it looks
weird, and b) it's a bit of an odd coupling with the `let` keyword as alternatives like this aren't
going to be introducing variable bindings. However, it would make `while let` more general and able
to replace more instances of `loop { match { ... } }` than is possible with the main design.

# Unresolved questions

None.
