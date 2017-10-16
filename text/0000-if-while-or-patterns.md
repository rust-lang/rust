- Feature Name: if_while_or_patterns
- Start Date: 2017-10-16
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Enables "or" patterns for [`if let`](https://github.com/rust-lang/rfcs/pull/160) and [`while let`](https://github.com/rust-lang/rfcs/pull/214) expressions. In other words, examples like the following are now possible:

```rust
enum E<T> {
    A(T), B(T), C, D, E, F
}

// Assume the enum E and the following for the remainder of the RFC:
use E::*;

let x = A(1);
let r = if let C | D = x { 1 } else { 2 };

while let A(x) | B(x) = source() {
    react_to(x);
}
```

# Motivation
[motivation]: #motivation

While nothing in this RFC is currently impossible in Rust, the changes the RFC proposes improves the ergonomics of control flow when dealing with `enum`s (sum types) with three or more variants where the program should react in one way to a group of variants, and another way to another group of variants. Examples of when such sum types occur are protocols and when dealing with languages (ASTs).

The following snippet (written with this RFC):

```rust
if let A(x) | B(x) = expr {
    do_stuff_with(x);
}
```

must be written as:

```rust
if let A(x) = expr {
    do_stuff_with(x);
} else if let B(x) = expr {
    do_stuff_with(x);
}
```

or, using `match`:

```rust
match expr {
    A(x) | B(x) => do_stuff_with(x),
    _           => {},
}
```

With `while let`, the ergonomics and in particular the readability can be significantly improved.

The following snippet (written with this RFC):

```rust
while let A(x) | B(x) = source() {
    react_to(x);
}
```

must currently be written as:

```rust
loop {
    match source() {
        A(x) | B(x) => react_to(x),
        _ => { break; }
    }
}
```

Another major motivation of the RFC is consistency with `match`.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

[RFC 2005](https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#examples), in describing the third example in the section "Examples", refers to patterns with `|` in them as "or" patterns. This RFC adopts the same terminology.

While the "sum" of all patterns in `match` must be irrefutable, or in other words: cover all cases, be exhaustive, this is not the case (currently) with `if/while let`, which may have a refutable pattern. This RFC does not change this.

The RFC only extends the use of or-patterns from `match`es to `if let` and `while let` expressions.

For examples, see [motivation].

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Grammar

### `if let`

The grammar in [§ 7.2.24](https://doc.rust-lang.org/grammar.html#if-let-expressions) is changed from:

```
if_let_expr : "if" "let" pat '=' expr '{' block '}'
               else_tail ? ;
```

to:

```
if_let_expr : "if" "let" pat [ '|' pat ] * '=' expr '{' block '}'
               else_tail ? ;
```

### `while let`

The grammar in [§ 7.2.25](https://doc.rust-lang.org/grammar.html#while-let-loops) is changed from:

```
while_let_expr : [ lifetime ':' ] ? "while" "let" pat '=' expr '{' block '}' ;
```

to:

```
while_let_expr : [ lifetime ':' ] ? "while" "let" pat [ '|' pat ] * '=' expr '{' block '}' ;
```

## Syntax lowering

The changes proposed in this RFC can be implemented by transforming the `if/while let` constructs with a syntax-lowering pass into `match` and `loop` + `match` expressions.

### Examples, `if let`

These examples are extensions on the [`if let` RFC](https://github.com/rust-lang/rfcs/pull/160). Therefore, the RFC avoids
duplicating any details already specified there.

Source:
```rust
if let PAT [| PAT]* = EXPR { BODY }
```
Result:
```rust
match EXPR {
    PAT [| PAT]* => { BODY }
    _ => {}
}
```

Source:
```rust
if let PAT [| PAT]* = EXPR { BODY_IF } else { BODY_ELSE }
```
Result:
```rust
match EXPR {
    PAT [| PAT]* => { BODY_IF }
    _ => { BODY_ELSE }
}
```

Source:
```rust
if COND {
    BODY_IF
} else if let PAT [| PAT]* = EXPR {
    BODY_ELSE_IF
} else {
    BODY_ELSE
}
```
Result:
```rust
if COND {
    BODY_IF
} else {
    match EXPR {
        PAT [| PAT]* => { BODY_ELSE_IF }
        _ => { BODY_ELSE }
    }
}
```

Source
```rust
if let PAT [| PAT]* = EXPR {
    BODY_IF
} else if COND {
    BODY_ELSE_IF_1
} else if OTHER_COND {
    BODY_ELSE_IF_2
}
```
Result:
```rust
match EXPR {
    PAT [| PAT]* => { BODY_IF }
    _ if COND => { BODY_ELSE_IF_1 }
    _ if OTHER_COND => { BODY_ELSE_IF_2 }
    _ => {}
}
```

### Examples, `while let`

The following example is an extension on the [`while let` RFC](https://github.com/rust-lang/rfcs/pull/214).

Source
```rust
['label:] while let PAT [| PAT]* = EXPR {
    BODY
}
```
Result:
```rust
['label:] loop {
    match EXPR {
        PAT [| PAT]* => BODY,
        _ => break
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

It's one more addition to the grammar.

# Rationale and alternatives
[alternatives]: #alternatives

This could simply not be done.
Consistency with `match` is however on its own reason enough to do this.

It could be claimed that the `if/while let` RFCs already mandate this RFC,
this RFC does answer that question and instead simply mandates it now.

# Unresolved questions
[unresolved]: #unresolved-questions

The exact syntax transformations should be deferred to the implementation.
This RFC does not mandate exactly how the AST:s should be transformed, only
that the or-pattern feature be supported.

There are no unresolved questions.