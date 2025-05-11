# `if let` guard in-depth review

Rust’s *`if let` guards* feature allows match arms to include an `if let` condition, enabling a pattern match and a conditional check
in one guard.  In other words, a match arm can be written as:

```rust
match value {
    PATTERN if let GUARD_PAT = GUARD_EXPR => { /* body */ },
    _ => { /* fallback */ },
}
```

This arm is selected **only if** `value` matches `PATTERN` *and* additionally `GUARD_EXPR` matches `GUARD_PAT`.  Crucially, the
variables bound by `PATTERN` are in scope when evaluating `GUARD_EXPR`, and the variables bound by both patterns
(`PATTERN` and `GUARD_PAT`) are in scope in the arm’s body.  For example:

```rust
match foo {
    Some(x) if let Ok(y) = compute(x) => {
        // `x` from `Some(x)` and `y` from `Ok(y)` are both available here
        println!("{}, {}", x, y);
    }
    _ => {}
}
```

As the original RFC explains, the semantics are that the guard is chosen *if* the main pattern matches and the let-pattern
in the guard also matches.  (Per design, a match arm may have *either* a normal `if` guard or an `if let` guard, but not both simultaneously.)

Currently the feature is **unstable** and gated by `#![feature(if_let_guard)]`.  If used without the feature, the compiler
emits error E0658: “`if let` guards are experimental” (with a suggestion to use `matches!(…)` instead).
Tests in the Rust repository (e.g. `feature-gate.rs`) verify that using `if let` in a guard without the feature flag
indeed produces this error.

## Syntax and Examples

The syntax for an `if let` guard follows the existing match-guard form, except using `if let` after the pattern:

```text
match EXPR {
    PAT if let P = E => BODY,
    // ...
}
```

Here `PAT` is an arbitrary pattern for the match arm, and `if let P = E` is the guard.  You can also combine multiple conditions
with `&&`.  In fact, because of the related “let chains” feature, you can write multiple `let`-bindings chained by `&&` in the
same guard.  For example:

```rust
match value {
    // Two let-conditions chained with `&&`
    (Some(a), Some(b)) if let Ok(x) = f(a) && let Ok(y) = g(b) => {
        // use a, b, x, y here
    }
    _ => {}
}
```

Examples of valid `if let` guards (with the feature enabled) include:

```rust
match x {
    (n, m) if let (0, Some(color)) = (n/10, color_for_code(m)) => { /* ... */ }
    y if let Some(z) = helper(y) => { /* ... */ }
    _ => { /* ... */ }
}
```

If the syntax is used incorrectly, the compiler gives an appropriate error. For instance, writing `(let PAT = EXPR)` parenthesized
or using `if (let PAT = EXPR)` (i.e. wrapping a `let` in extra parentheses) is not accepted as a valid guard and instead produces
a parse error “expected expression, found `let` statement”. This is tested in the Rust UI tests (e.g. `parens.rs` and `feature-gate.rs`). In short, `if let` must appear exactly as a guard after an `if`, not inside extra parentheses.

## Semantics and Variable Scope

When a match arm has an `if let` guard, the evaluation proceeds as follows:

1. The match scrutinee is matched against the arm’s main pattern `PAT`.  Any variables bound by `PAT` become available.
2. If the main pattern matches, then the guard expression is evaluated.  In that expression, the bindings from `PAT` can be used.  The guard expression is of the form `let GUARD_PAT = GUARD_EXPR`.
3. The result of `GUARD_EXPR` is matched against `GUARD_PAT`.  If this succeeds, then execution enters the arm’s body.  Otherwise the arm is skipped (and later arms are tried).

Therefore, variables bound in the main pattern `PAT` are “live” during the evaluation of the guard, but any variables bound
by `GUARD_PAT` only come into existence in the arm body (not in earlier code).  This corresponds directly to the RFC’s reference
explanation: “the variables of `pat` are bound in `guard_expr`, and the variables of `pat` and `guard_pat` are bound in `body_expr`”.

As an example, consider:

```rust
match (opt, val) {
    (Some(x), _) if let Ok(y) = convert(x) => {
        // Here `x` and `y` are in scope
        println!("Converted {} into {}", x, y);
    }
    _ => {}
}
```

Here the pattern `(Some(x), _)` binds `x`. Then `convert(x)` is called, and its result is matched to `Ok(y)`.  If both steps
succeed, the body can use both `x` and `y`.  If either fails (pattern fails or guard fails), this arm is skipped.

One important restriction is that a single match arm cannot have two `if`-guards.  That is, you cannot write something like
`PAT if cond1 if let P = E => ...` with two separate `if`s.  You may combine a normal boolean condition with a `let`
by chaining with `&&`, but only one `if` keyword is allowed.  The RFC explicitly states “An arm may not have both an
`if` and an `if let` guard” (i.e. you can’t do `if cond && let ...` *and* then another `if`, etc.).
(You *can* do something like `if let P = E && cond` by writing `if let P = E && cond =>`, treating the boolean as part
of a let-chain, but that is a single `if` in syntax.)

## Feature Gate and Errors

As of now, `if let` guards are still unstable. The compiler requires the feature flag `#![feature(if_let_guard)]` to enable them.
If one uses an `if let` guard without the feature, one gets an error similar to:

```
error[E0658]: `if let` guards are experimental
   |
LL |     _ if let true = true => {}
   |        ^^^^^^^^^^^^^^^^
   = help: you can write `if matches!(<expr>, <pattern>)` instead of `if let <pattern> = <expr>`
```

This message is verified by the compiler’s test suite (e.g. `feature-gate.rs`) and comes from the feature-gate code in the parser.
The tests also ensure the old (`let`-in-`if` without the feature) error is preserved. For example:

```rust
match () {
    () if true && let 0 = 1 => {}      // error: `let` expressions are unstable (since no feature)
    () if let 0 = 1 && true => {}      // error: `if let` guards are experimental
    _ => {}
}
```

The test suite checks that these errors mention both the unstable-let and the experimental guard exactly as above.
Once the feature is stabilized, these errors will no longer appear.

## Temporaries and Drop Order

A subtle aspect of `if let` guards is the handling of temporaries (and destructor calls) within the guard expression.
The Rust reference explains that a *match guard* creates its own temporary scope: any temporaries produced by `GUARD_EXPR`
live only until the guard finishes evaluating. Concretely, this means:

* The `guard_expr` is evaluated *after* matching `PAT` but *before* executing the arm’s body (if taken).
* Any temporary values created during `guard_expr` are dropped immediately after the guard’s scope ends (i.e. before entering the arm body).
* If the guard fails, those temporaries are dropped right then, and the compiler proceeds to the next arm.

In effect, the drop semantics are the same as for an ordinary match guard or an `if let` in an `if` expression: no unexpected
extension of lifetimes. (In Rust 2024 edition, there is a finer rule that even in `if let`
expressions temporaries drop before the `else` block; but for match guards the effect is that temporaries from the
guard are dropped before the arm body.)

This behavior is exercised by the existing tests. For example, the `drop-order.rs` UI test uses `Drop`-implementing
values in nested `if let` guards to verify the precise drop order. Those tests confirm that the values from the inner
guards are dropped *first*, before values from outer contexts and before finally moving on to other arms. In short, the
feature does not introduce any new irregularity in drop order: guard expressions are evaluated left-to-right
(following let-chains semantics) and their temporaries die as soon as the guard completes.

## Lifetimes and Variable Scope

Aside from drop timing, lifetimes of references in the guard work as expected. Because the pattern variables (`PAT` bindings)
are in scope during `GUARD_EXPR`, one can take references to them or otherwise use them. Any reference or borrow introduced
by the guard is scoped to the guard and arm body. For example:

```rust
match &vec {
    v if let [first, ref rest @ ..] = v[..] => {
        // `first` and `rest` borrowed from `v` are valid here
        println!("{}", first);
    }
    _ => {}
}
```

Here `v` is `&Vec`, and the guard borrows parts of it; those references are valid in the arm body. If a guard binds by value
(e.g. `if let x = some_moveable`), the usual move/borrow rules apply (see below), but in all cases the scopes follow the match-arm rules.

Moreover, an `if let` guard cannot break exhaustiveness: each arm is either taken or skipped in the usual way.
A guard cannot cause a pattern to match something it wouldn’t normally match, it only *restricts* a match further.
Tests like `exhaustive.rs` ensure that match exhaustiveness is checked as usual (you still need a wildcard arm if needed).
No special exhaustiveness rules are introduced.

## Mutability and Moves

Patterns inside guards obey the normal mutability and move semantics. You can use `mut`, `ref`, or `ref mut`
in the guard pattern just like in a `let` or match pattern. For example, `if let Some(ref mut x) = foo()` will mutably
borrow from `foo()`. The borrow-checker treats moves in a guard pattern exactly as it would in a regular pattern: a move of a
binding only occurs if that branch is actually taken, and subsequent code cannot use a moved value.

This is tested by the **move-guard-if-let** suite.  For instance, consider:

```rust
fn same_pattern(c: bool) {
    let mut x: Box<_> = Box::new(1);
    let v = (1, 2);
    match v {
        (1, _) if let y = *x && c => (),
        (_, 2) if let z = *x => (),     // uses x after move
        _ => {}
    }
}
```

With `#![feature(if_let_guard)]`, the compiler correctly reports that `x` is moved by the first guard and then used again by
the second pattern, which is an error. In the test output one sees messages like “`value moved here`”
and “`value used here after move`” exactly pointing to the `if let` bindings. (These errors match the compiler’s normal behavior,
confirming that `if let` guards do not bypass the borrow rules.) In contrast, if the pattern had used `ref` (e.g. `if let ref y = x`),
no move would occur. The test suite also covers using `&&` with or-patterns and ensures borrowck handles those correctly.

In summary, moving or borrowing in an `if let` guard is just like doing so in a regular `if let` or match: the borrow checker
ensures no use-after-move, and moves only happen if the pattern actually matches. The existing UI tests for moves and mutability
all pass under the current implementation, so there is no unsoundness here.

## Shadowing and Macros

The usual Rust rules for shadowing and macros apply. An `if let` guard can introduce a new variable that *shadows* an existing one:

```rust
let x = 10;
match v {
    (true, _) if let x = compute() => {
        // Here the `x` from the guard shadows the outer `x`.
        println!("{}", x);
    }
    _ => {}
}
```

This is allowed (just as in ordinary `if let` expressions) and works as expected; the tests (`shadowing.rs`) verify that the
scoping is consistent.

Macro expansion also works naturally. You can write macros that produce part of the guard. For example:

```rust
macro_rules! m { () => { Some(5) } }
match opt {
    Some(v) if let Some(w) = m!() => { /*...*/ }
    _ => {}
}
```

Since the parser sees the expanded code, `if let` guards inside macros are supported. The Rust tests include cases where macros
expand to an `if let` guard (fully or partially) to ensure the feature handles macro hygiene correctly. In short, `if let` guards
are not disabled or altered in macro contexts; they simply follow the normal macro expansion rules of Rust.

## Soundness and Pitfalls

No inherent unsoundness has been found in `if let` guards. They are purely syntactic sugar for nested pattern matching and condition
checks. All borrow and move checks are done conservatively and correctly. The feature interacts with other parts of the language in
predictable ways. For example:

* **Refutability:**  An `if let` guard’s pattern is allowed to be refutable (since a failed match simply means skipping the arm). The tests ensure that irrefutable-let warnings do not occur (or can be allowed).
* **Matching order:**  Guards are evaluated in sequence per arm; if the first part of a let-chain fails, later parts aren’t evaluated (preventing needless moves or panics).
* **No new invariants:** Guard patterns do not introduce new lifetime or aliasing invariants beyond normal patterns. Temporaries and borrows expire normally.

All of the edge cases are covered by the existing UI tests. For example, the `exhaustive.rs` test confirms that match exhaustiveness
remains correct when using `if let` guards (i.e. a wildcard arm is needed if not all cases are covered).
The `typeck.rs` and `type-inference.rs` tests verify that type inference and generic code work through `if let` guards as expected.
The compiler’s own test suite includes dozens of `if let` guard tests under `src/test/ui/rfcs/rfc-2294-if-let-guard/`,
and all of them pass with the current implementation.

## Conclusion

The feature is fully implemented in the compiler and exercised by many tests. Its syntax and semantics are clear and consistent with
existing Rust rules: pattern bindings from the arm are in scope in the guard, and guard bindings are in scope in the arm body.
The compiler enforces the usual ownership rules (as seen in the move tests) and handles temporaries in a straightforward way.

**Status:** implemented and well-tested, awaiting only formal documentation [(I've also made one)](https://github.com/rust-lang/reference/pull/1823) to be fully ready for a stable release.

**References:** details from RFC 2294 and the current compiler behavior are used above. Each cited source shows the design or
diagnostics of `if let` guards in action.
