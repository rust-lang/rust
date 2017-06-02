- Feature Name: generic_assert
- Start Date: 2017-05-24
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Make the `assert!` macro generic to all expressions, and extend the readability of debug dumps.

# Motivation
[motivation]: #motivation

While clippy warns about `assert!` usage that should be replaced by `assert_eq!`, it's quite annoying to migrating around.

Unit test frameworks like [Catch](https://github.com/philsquared/Catch) for C++ does cool message printing already by using macros.

# Detailed design
[design]: #detailed-design

We're going to parse AST and break up them by operators (excluding `.` (dot, member access operator)). Function calls and bracket surrounded blocks are considered as one block and don't get split further.

On assertion failure, the expression itself is stringified, and another line with intermediate values are printed out. The values should be printed with `Debug`, and a plain text fallback if the following conditions fail:
- the type doesn't implement `Debug`.
- the operator is non-comparison (those in `std::ops`) and the type (may also be a reference) doesn't implement `Copy`.

To make sure that there's no side effects involved (e.g. running `next()` twice on `Iterator`), each value should be stored as temporaries and dumped on assertion failure.

## Examples

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
- This has a potential performance degradation on complex expressions, due to creating more temporaries on stack (or register).

# Alternatives
[alternatives]: #alternatives

- Defining via `macro_rules!` was considered, but the recursive macro can often reach the recursion limit.

# Unresolved questions
[unresolved]: #unresolved-questions

## Error messages
- Should we use the negated version of operators (e.g. `!=` for eq) to explain the failure?
- Should we dump the AST as a formatted one?
- How are we going to handle multi-line expressions?

## Operators
- Should we handle non-comparison operators?
