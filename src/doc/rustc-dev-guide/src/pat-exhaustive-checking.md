# Pattern and exhaustiveness checking

In Rust, pattern matching and bindings have a few very helpful properties. The
compiler will check that bindings are irrefutable when made and that match arms
are exhaustive.

## Pattern usefulness

The central question that usefulness checking answers is:
"in this match expression, is that branch redundant?".
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

This check is done anywhere you can write a pattern: `match` expressions, `if let`, `let else`,
plain `let`, and function arguments.

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

Exhaustiveness checking is run before MIR building in [`check_match`].
It is implemented in the [`rustc_pattern_analysis`] crate,
with the core of the algorithm in the [`usefulness`] module.
That file contains a detailed description of the algorithm.

## Important concepts

### Constructors and fields

In the value `Pair(Some(0), true)`, `Pair` is called the constructor of the value, and `Some(0)` and
`true` are its fields. Every matchable value can be decomposed in this way. Examples of
constructors are: `Some`, `None`, `(,)` (the 2-tuple constructor), `Foo {..}` (the constructor for
a struct `Foo`), and `2` (the constructor for the number `2`).

Each constructor takes a fixed number of fields; this is called its arity. `Pair` and `(,)` have
arity 2, `Some` has arity 1, `None` and `42` have arity 0. Each type has a known set of
constructors. Some types have many constructors (like `u64`) or even an infinitely many (like `&str`
and `&[T]`).

Patterns are similar: `Pair(Some(_), _)` has constructor `Pair` and two fields. The difference is
that we get some extra pattern-only constructors, namely: the wildcard `_`, variable bindings,
integer ranges like `0..=10`, and variable-length slices like `[_, .., _]`. We treat or-patterns
separately.

Now to check if a value `v` matches a pattern `p`, we check if `v`'s constructor matches `p`'s
constructor, then recursively compare their fields if necessary. A few representative examples:

- `matches!(v, _) := true`
- `matches!((v0,  v1), (p0,  p1)) := matches!(v0, p0) && matches!(v1, p1)`
- `matches!(Foo { a: v0, b: v1 }, Foo { a: p0, b: p1 }) := matches!(v0, p0) && matches!(v1, p1)`
- `matches!(Ok(v0), Ok(p0)) := matches!(v0, p0)`
- `matches!(Ok(v0), Err(p0)) := false` (incompatible variants)
- `matches!(v, 1..=100) := matches!(v, 1) || ... || matches!(v, 100)`
- `matches!([v0], [p0, .., p1]) := false` (incompatible lengths)
- `matches!([v0, v1, v2], [p0, .., p1]) := matches!(v0, p0) && matches!(v2, p1)`

This concept is absolutely central to pattern analysis. The [`constructor`] module provides
functions to extract, list and manipulate constructors. This is a useful enough concept that
variations of it can be found in other places of the compiler, like in the MIR-lowering of a match
expression and in some clippy lints.

### Constructor grouping and splitting

The pattern-only constructors (`_`, ranges and variable-length slices) each stand for a set of
normal constructors, e.g. `_: Option<T>` stands for the set {`None`, `Some`} and `[_, .., _]` stands
for the infinite set {`[,]`, `[,,]`, `[,,,]`, ...} of the slice constructors of arity >= 2.

In order to manage these constructors, we keep them as grouped as possible. For example:

```rust
match (0, false) {
    (0 ..=100, true) => {}
    (50..=150, false) => {}
    (0 ..=200, _) => {}
}
```

In this example, all of `0`, `1`, .., `49` match the same arms, and thus can be treated as a group.
In fact, in this match, the only ranges we need to consider are: `0..50`, `50..=100`,
`101..=150`,`151..=200` and `201..`. Similarly:

```rust
enum Direction { North, South, East, West }
# let wind = (Direction::North, 0u8);
match wind {
    (Direction::North, 50..) => {}
    (_, _) => {}
}
```

Here we can treat all the non-`North` constructors as a group, giving us only two cases to handle:
`North`, and everything else.

This is called "constructor splitting" and is crucial to having exhaustiveness run in reasonable
time.

### Usefulness vs reachability in the presence of empty types

This is likely the subtlest aspect of exhaustiveness. To be fully precise, a match doesn't operate
on a value, it operates on a place. In certain unsafe circumstances, it is possible for a place to
not contain valid data for its type. This has subtle consequences for empty types. Take the
following:

```rust
enum Void {}
let x: u8 = 0;
let ptr: *const Void = &x as *const u8 as *const Void;
unsafe {
    match *ptr {
        _ => println!("Reachable!"),
    }
}
```

In this example, `ptr` is a valid pointer pointing to a place with invalid data. The `_` pattern
does not look at the contents of the place `*ptr`, so this code is ok and the arm is taken. In other
words, despite the place we are inspecting being of type `Void`, there is a reachable arm. If the
arm had a binding however:

```rust
# #[derive(Copy, Clone)]
# enum Void {}
# let x: u8 = 0;
# let ptr: *const Void = &x as *const u8 as *const Void;
# unsafe {
match *ptr {
    _a => println!("Unreachable!"),
}
# }
```

Here the binding loads the value of type `Void` from the `*ptr` place. In this example, this causes
UB since the data is not valid. In the general case, this asserts validity of the data at `*ptr`.
Either way, this arm will never be taken.

Finally, let's consider the empty match `match *ptr {}`. If we consider this exhaustive, then
having invalid data at `*ptr` is invalid. In other words, the empty match is semantically
equivalent to the `_a => ...` match. In the interest of explicitness, we prefer the case with an
arm, hence we won't tell the user to remove the `_a` arm. In other words, the `_a` arm is
unreachable yet not redundant. This is why we lint on redundant arms rather than unreachable
arms, despite the fact that the lint says "unreachable".

These considerations only affects certain places, namely those that can contain non-valid data
without UB. These are: pointer dereferences, reference dereferences, and union field accesses. We
track during exhaustiveness checking whether a given place is known to contain valid data.

Having said all that, the current implementation of exhaustiveness checking does not follow the
above considerations. On stable, empty types are for the most part treated as non-empty. The
[`exhaustive_patterns`] feature errs on the other end: it allows omitting arms that could be
reachable in unsafe situations. The [`never_patterns`] experimental feature aims to fix this and
permit the correct behavior of empty types in patterns.

[`check_match`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/pattern/check_match/index.html
[`rustc_pattern_analysis`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_pattern_analysis/index.html
[`usefulness`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_pattern_analysis/usefulness/index.html
[`constructor`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_pattern_analysis/constructor/index.html
[`never_patterns`]: https://github.com/rust-lang/rust/issues/118155
[`exhaustive_patterns`]: https://github.com/rust-lang/rust/issues/51085
