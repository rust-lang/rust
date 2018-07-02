- Feature Name: `const_looping`
- Start Date: 2018-02-18
- RFC PR: [rust-lang/rfcs#2344](https://github.com/rust-lang/rfcs/pull/2344)
- Rust Issue: [rust-lang/rust#52000](https://github.com/rust-lang/rust/issues/52000)

# Summary
[summary]: #summary

Allow the use of `loop`, `while` and `while let` during constant evaluation.
`for` loops are technically allowed, too, but can't be used in practice because
each iteration calls `iterator.next()`, which is not a `const fn` and thus can't
be called within constants. Future RFCs (like
https://github.com/rust-lang/rfcs/pull/2237) might lift that restriction.

# Motivation
[motivation]: #motivation

Any iteration is expressible with recursion. Since we already allow recursion
via const fn and termination of said recursion via `if` or `match`, all code
enabled by const recursion is already legal now. Some algorithms are better
expressed as imperative loops and a lot of Rust code uses loops instead of
recursion. Allowing loops in constants will allow more functions to become const
fn without requiring any changes.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

If you previously had to write functional code inside constants, you can now
change it to imperative code. For example if you wrote a fibonacci like

```rust
const fn fib(n: u128) -> u128 {
    match n {
        0 => 1,
        1 => 1,
        n => fib(n - 1) + fib(n + 1)
    }
}
```

which takes exponential time to compute a fibonacci number, you could have
changed it to the functional loop

```rust
const fn fib(n: u128) -> u128 {
    const fn helper(n: u128, a: u128, b: u128, i: u128) -> u128 {
        if i <= n {
            helper(n, b, a + b, i + 1)
        } else {
            b
        }
    }
    helper(n, 1, 1, 2)
}
```

but now you can just write it as an imperative loop, which also finishes in
linear time.

```rust
const fn fib(n: u128) -> u128 {
    let mut a = 1;
    let mut b = 1;
    let mut i = 2;
    while i <= n {
        let tmp = a + b;
        a = b;
        b = tmp;
        i += 1;
    }
    b
}
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

A loop in MIR is a cyclic graph of `BasicBlock`s. Evaluating such a loop is no
different from evaluating a linear sequence of `BasicBlock`s, except that
termination is not guaranteed. To ensure that the compiler never hangs
indefinitely, we count the number of terminators processed and whenever we reach
a fixed limit, we report a lint mentioning that we cannot guarantee that the
evaluation will terminate and reset the counter to zero. This lint should recur
in a non-annoying amount of time (e.g. at least 30 seconds between occurrences).
This means that there's an internal deterministic counter (for the terminators) and
a timestamp of the last (if any) loop warning emission. Both the counter needs to reach
its limit and 30 seconds have to have passed since the last warning emission in order
for a new warning to be emitted.

# Drawbacks
[drawbacks]: #drawbacks

* Infinite loops will cause the compiler to never finish if the lint is not denied

# Rationale and alternatives
[alternatives]: #alternatives

- Do nothing, users can keep using recursion

# Unresolved questions
[unresolved]: #unresolved-questions

* Should we add a true recursion check that hashes the interpreter state and
  detects if it has reached the same state again?
    * This will slow down const evaluation enormously and for complex iterations
      is essentially useless because it'll take forever (e.g. counting from 0 to
      `u64::max_value()`)
