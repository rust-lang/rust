- Start Date: 2014-06-05
- RFC PR: [rust-lang/rfcs#107](https://github.com/rust-lang/rfcs/pull/107)
- Rust Issue: [rust-lang/rust#15287](https://github.com/rust-lang/rust/issues/15287)

# Summary

Rust currently forbids pattern guards on match arms with move-bound variables.
Allowing them would increase the applicability of pattern guards.

# Motivation

Currently, if you attempt to use guards on a match arm with a move-bound
variable, e.g.

```rust
struct A { a: Box<int> }

fn foo(n: int) {
    let x = A { a: box n };
    let y = match x {
        A { a: v } if *v == 42 => v,
        _ => box 0
    };
}
```

you get an error:

```
test.rs:6:16: 6:17 error: cannot bind by-move into a pattern guard
test.rs:6         A { a: v } if *v == 42 => v,
                         ^
```

This should be permitted in cases where the guard only accesses the moved value
by reference or copies out of derived paths.

This allows for succinct code with less pattern matching duplication and a
minimum number of copies at runtime. The lack of this feature was encountered by
@kmcallister when developing Servo's new HTML 5 parser.

# Detailed design

This change requires all occurrences of move-bound pattern variables in the
guard to be treated as paths to the values being matched before they are moved,
rather than the moved values themselves. Any moves of matched values into the
bound variables would occur on the control flow edge between the guard and the
arm's expression. There would be no changes to the handling of reference-bound
pattern variables.

The arm would be treated as its own nested scope with respect to borrows, so
that pattern-bound variables would be able to be borrowed and dereferenced
freely in the guard, but these borrows would not be in scope in the arm's
expression. Since the guard dominates the expression and the move into the
pattern-bound variable, moves of either the match's head expression or any
pattern-bound variables in the guard would trigger an error.

The following examples would be accepted:

```rust
struct A { a: Box<int> }

impl A {
    fn get(&self) -> int { *self.a }
}

fn foo(n: int) {
    let x = A { a: box n };
    let y = match x {
        A { a: v } if *v == 42 => v,
        _ => box 0
    };
}

fn bar(n: int) {
    let x = A { a: box n };
    let y = match x {
        A { a: v } if x.get() == 42 => v,
        _ => box 0
    };
}

fn baz(n: int) {
    let x = A { a: box n };
    let y = match x {
        A { a: v } if *v.clone() == 42 => v,
        _ => box 0
    };
}
```

This example would be rejected, due to a double move of `v`:

```rust
struct A { a: Box<int> }

fn foo(n: int) {
    let x = A { a: box n };
    let y = match x {
        A { a: v } if { drop(v); true } => v,
        _ => box 0
    };
}
```

This example would also be rejected, even though there is no use of the
move-bound variable in the first arm's expression, since the move into the bound
variable would be moving the same value a second time:

```rust
enum VecWrapper { A(Vec<int>) }

fn foo(x: VecWrapper) -> uint {
    match x {
        A(v) if { drop(v); false } => 1,
        A(v) => v.len()
    }
}
```

There are issues with mutation of the bound values, but that is true without
the changes proposed by this RFC, e.g.
[Rust issue #14684](https://github.com/mozilla/rust/issues/14684). The general
approach to resolving that issue should also work with these proposed changes.

This would be implemented behind a `feature(bind_by_move_pattern_guards)` gate
until we have enough experience with the feature to remove the feature gate.

# Drawbacks

The current error message makes it more clear what the user is doing wrong, but
if this change is made the error message for an invalid use of this feature
(even if it were accidental) would indicate a use of a moved value, which might
be more confusing.

This might be moderately difficult to implement in `rustc`.

# Alternatives

As far as I am aware, the only workarounds for the lack of this feature are to
manually expand the control flow of the guard (which can quickly get messy) or
use unnecessary copies.

# Unresolved questions

This has nontrivial interaction with guards in arbitrary patterns as proposed
in [#99](https://github.com/rust-lang/rfcs/pull/99).

