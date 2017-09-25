- Feature Name: `generic_assert`
- Start Date: 2017-05-24
- RFC PR: [rust-lang/rfcs#2011](https://github.com/rust-lang/rfcs/pull/2011)
- Rust Issue: [rust-lang/rust#44838](https://github.com/rust-lang/rust/issues/44838)

# Summary
[summary]: #summary

Make the `assert!` macro recognize more expressions (utilizing the power of procedural macros), and extend the readability of debug dumps.

# Motivation
[motivation]: #motivation

While clippy warns about `assert!` usage that should be replaced by `assert_eq!`, it's quite annoying to migrate around.

Unit test frameworks like [Catch](https://github.com/philsquared/Catch) for C++ does cool message printing already by using macros.

# Detailed design
[design]: #detailed-design

We're going to parse AST and break up them by operators (excluding `.` (dot, member access operator)). Function calls and bracket surrounded blocks are considered as one block and don't get expanded. The exact expanding rules should be determined when implemented, but an example is provided for reference.

On assertion failure, the expression itself is stringified, and another line with intermediate values are printed out. The values should be printed with `Debug`, and a plain text fallback if the following conditions fail:
- the type doesn't implement `Debug`.
- the operator is non-comparison (those in `std::ops`) and the type (may also be a reference) doesn't implement `Copy`.

To make sure that there's no side effects involved (e.g. running `next()` twice on `Iterator`), each value should be stored as temporaries and dumped on assertion failure.

The new assert messages are likely to generate longer code, and it may be simplified for release builds (if benchmarks confirm the slowdown).

## Examples

These examples are purely for reference. The implementor is free to change the rules.

```rust
let a = 1;
let b = 2;
assert!(a == b);
```

```
thread '<main>' panicked at 'assertion failed:
Expected: a == b
With expansion: 1 == 2'
```

With addition operators:

```rust
let a = 1;
let b = 1;
let c = 3;
assert!(a + b == c);
```

```
thread '<main>' panicked at 'assertion failed:
Expected: a + b == c
With expansion: 1 + 1 == 3'
```

Bool only:
```rust
let v = vec![0u8;1];
assert!(v.is_empty());
```

```
thread '<main>' panicked at 'assertion failed:
Expected: v.is_empty()'
```

With short-circuit:
```rust
assert!(true && false && true);
```

```
thread '<main>' panicked at 'assertion failed:
Expected: true && false && true
With expansion: true && false && (not evaluated)'
```

With bracket blocks:
```rust
let a = 1;
let b = 1;
let c = 3;
assert!({a + b} == c);
```

```
thread '<main>' panicked at 'assertion failed:
Expected: {a + b} == c
With expansion: 2 == 3'
```

With fallback:
```rust
let a = NonDebug{};
let b = NonDebug{};
assert!(a == b);
```
```
thread '<main>' panicked at 'assertion failed:
Expected: a == b
With expansion: (a) == (b)'
```


# How We Teach This
[how-we-teach-this]: #how-we-teach-this

- Port the documentation (and optionally compiler source) to use `assert!`.
- Mark the old macros (`assert_{eq,ne}!`) as deprecated.

# Drawbacks
[drawbacks]: #drawbacks

- This will generate a wave of deprecation warnings, which will be some cost for users to migrate. However, this doesn't mean that this is backward-incompatible, as long as the deprecated macros aren't removed.
- This has a potential performance degradation on complex expressions, due to creating more temporaries on stack (or register). However, if this had clear impacts confirmed through benchmarks, we should use some kind of alternative implementation for release builds.

# Alternatives
[alternatives]: #alternatives

- Defining via `macro_rules!` was considered, but the recursive macro can often reach the recursion limit.
- Negating the operator (`!=` to `==`) was considered, but this isn't suitable for all cases as not all types are total ordering.

# Unresolved questions
[unresolved]: #unresolved-questions

These questions should be settled during the implementation process.

## Error messages
- Should we dump the AST as a formatted one?
- How are we going to handle multi-line expressions?

## Operators
- Should we handle non-comparison operators?
