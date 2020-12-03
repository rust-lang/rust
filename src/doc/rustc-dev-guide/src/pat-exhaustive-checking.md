# Pattern and Exhaustiveness Checking

In Rust, pattern matching and bindings have a few very helpful properties. The
compiler will check that bindings are irrefutable when made and that match arms
are exhaustive.

## Pattern usefulness

The central question that usefulness checking answers is:
"in this match expression, is that branch reachable?".
More precisely, it boils down to computing whether,
given a list of patterns we have already seen,
a given new pattern might match any new value.

For example, in the following match expression,
we ask in turn whether each pattern might match something
that wasn't matched by the patterns above it.
Here we see the 4th pattern is redundant with the 1st;
that branch will get an "unreachable" warning.
The 3rd pattern may or may not be useful,
depending on whether `Foo` has other variants than `Bar`.
Finally, we can ask whether the whole match is exhaustive
by asking whether the wildcard pattern (`_`)
is useful relative to the list of all the patterns in that match.
Here we can see that `_` is useful (it would catch `(false, None)`);
this expression would therefore get a "non-exhaustive match" error.

```rust
// x: (bool, Option<Foo>)
match x {
    (true, _) => {} // 1
    (false, Some(Foo::Bar)) => {} // 2
    (false, Some(_)) => {} // 3
    (true, None) => {} // 4
}
```

Thus usefulness is used for two purposes:
detecting unreachable code (which is useful to the user),
and ensuring that matches are exhaustive (which is important for soundness,
because a match expression can return a value).

## Where it happens

This check is done to any expression that desugars to a match expression in MIR.
That includes actual `match` expressions,
but also anything that looks like pattern matching,
including `if let`, destructuring `let`, and similar expressions.

```rust
// `match`
// Usefulness can detect unreachable branches and forbid non-exhaustive matches.
match foo() {
    Ok(x) => x,
    Err(_) => panic!(),
}

// `if let`
// Usefulness can detect unreachable branches.
if let Some(x) = foo() {
    // ...
}

// `while let`
// Usefulness can detect infinite loops and dead loops.
while let Some(x) = it.next() {
    // ...
}

// Destructuring `let`
// Usefulness can forbid non-exhaustive patterns.
let Foo::Bar(x, y) = foo();

// Destructuring function arguments
// Usefulness can forbid non-exhaustive patterns.
fn foo(Foo { x, y }: Foo) {
    // ...
}
```

## The algorithm

Exhaustiveness checking is implemented in [`check_match`].
The core of the algorithm is in [`usefulness`].
That file contains a detailed description of the algorithm.

[`check_match`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/pattern/check_match/index.html
[`usefulness`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/pattern/usefulness/index.html
