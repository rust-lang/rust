- Feature Name: `const-control-flow`
- Start Date: 2018-01-11
- RFC PR: [rust-lang/rfcs#2342](https://github.com/rust-lang/rfcs/pull/2342)
- Rust Issue: [rust-lang/rust#49146](https://github.com/rust-lang/rust/issues/49146)

# Summary
[summary]: #summary

Enable `if` and `match` during const evaluation and make them evaluate lazily.
In short, this will allow `if x < y { y - x } else { x - y }` even though the
else branch would emit an overflow error for unsigned types if `x < y`.

# Motivation
[motivation]: #motivation

Conditions in constants are important for making functions like `NonZero::new`
const fn and interpreting assertions.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

If you write

```rust
let x: u32 = ...;
let y: u32 = ...;
let a = x - y;
let b = y - x;
if x > y {
    // do something with a
} else {
    // do something with b
}
```

The program will always panic (except if both `x` and `y` are `0`) because
either `x - y` will overflow or `y - x` will. To resolve this one must move the
`let a` and `let b` into the `if` and `else` branch respectively.

```rust
let x: u32 = ...;
let y: u32 = ...;
if x > y {
    let a = x - y;
    // do something with a
} else {
    let b = y - x;
    // do something with b
}
```

When constants are involved, new issues arise:

```rust
const X: u32 = ...;
const Y: u32 = ...;
const FOO: SomeType = if X > Y {
    const A: u32 = X - Y;
    ...
} else {
    const B: u32 = Y - X;
    ...
};
```

`A` and `B` are evaluated before `FOO`, since constants are by definition
constant, so their order of evaluation should not matter. This assumption breaks
in the presence of errors, because errors are side effects, and thus not pure.

To resolve this issue, one needs to eliminate the intermediate constants and
directly evaluate `X - Y` and `Y - X`.

```rust
const X: u32 = ...;
const Y: u32 = ...;
const FOO: SomeType = if X > Y {
    let a = X - Y;
    ...
} else {
    let b = Y - X;
    ...
};
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

`match` on enums whose variants have no fields or `if` is translated during HIR
-> MIR lowering to a `switchInt` terminator. Mir interpretation will now have to
evaluate those terminators (which it already can).

`match` on enums with variants which have fields is translated to `switch`,
which will check either the discriminant or compute the discriminant in the case
of packed enums like `Option<&T>` (which has no special memory location for the
discriminant, but encodes `None` as all zeros and treats everything else as a
`Some`). When entering a `match` arm's branch, the matched on value is
essentially transmuted to the enum variant's type, allowing further code to
access its fields.

# Drawbacks
[drawbacks]: #drawbacks

This makes it easier to fail compilation on random "constant" values like
`size_of::<T>()` or other platform specific constants.

# Rationale and alternatives
[alternatives]: #alternatives

## Require intermediate const fns to break the eager const evaluation

Instead of writing

```rust
const X: u32 = ...;
const Y: u32 = ...;
const AB: u32 = if X > Y {
    X - Y
} else {
    Y - X
};
```

where either `X - Y` or `Y - X` would emit an error, add an intermediate const fn

```rust
const X: u32 = ...;
const Y: u32 = ...;
const fn foo(x: u32, y: u32) -> u32 {
    if x > y {
        x - y
    } else {
        y - x
    }
}
const AB: u32 = foo(X, Y);
```

Since the const fn's `x` and `y` arguments are unknown, they cannot be const
evaluated. When the const fn is evaluated with given arguments, only the taken
branch is evaluated.

# Unresolved questions
[unresolved]: #unresolved-questions
