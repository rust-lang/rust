- Feature Name: if_while_or_patterns
- Start Date: 2017-10-16
- RFC PR: [rust-lang/rfcs#2175](https://github.com/rust-lang/rfcs/pull/2175)
- Rust Issue: [rust-lang/rust#48215](https://github.com/rust-lang/rust/issues/48215)

# Summary
[summary]: #summary

[`if let`]: https://github.com/rust-lang/rfcs/pull/160
[`while let`]: https://github.com/rust-lang/rfcs/pull/214

Enables "or" patterns for [`if let`] and [`while let`] expressions as well as
`let` statements. In other words, examples like the following are now possible:

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

enum ParameterKind<T, L = T> { Ty(T), Lifetime(L), }

// Only possible when `L = T` such that `kind : ParameterKind<T, T>`.
let Ty(x) | Lifetime(x) = kind;
```

# Motivation
[motivation]: #motivation

While nothing in this RFC is currently impossible in Rust, the changes the RFC
proposes improves the ergonomics of control flow when dealing with `enum`s
(sum types) with three or more variants where the program should react in one
way to a group of variants, and another way to another group of variants.
Examples of when such sum types occur are protocols, when dealing with
languages (ASTs), and non-trivial iterators.

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

[`std::iter`]: https://doc.rust-lang.org/nightly/src/core/iter/mod.rs.html#691

This way of using `match` is seen multiple times in [`std::iter`] when dealing
with the `Chain` iterator adapter. An example of this is:

```rust
    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        match self.state {
            ChainState::Both | ChainState::Front => {
                accum = self.a.fold(accum, &mut f);
            }
            _ => { }
        }
        match self.state {
            ChainState::Both | ChainState::Back => {
                accum = self.b.fold(accum, &mut f);
            }
            _ => { }
        }
        accum
    }
```

which could have been written as:

```rust
    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc,
    {
        use ChainState::*;
        let mut accum = init;
        if let Both | Front = self.state { accum = self.a.fold(accum, &mut f); }
        if let Both | Back  = self.state { accum = self.b.fold(accum, &mut f); }
        accum
    }
```

This version is both shorter and clearer.

With `while let`, the ergonomics and in particular the readability can be
significantly improved.

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

To keep `let` statements consistent with `if let`, and to enable the scenario
exemplified by `ParameterKind` in the [motivation], these or-patterns are
allowed at the top level of `let` statements.

In addition to the `ParameterKind` example, we can also consider
`slice.binary_search(&x)`. If we are only interested in the `index` at where
`x` is or would be, without any regard for if it was there or not, we can
now simply write:

```rust
let Ok(index) | Err(index) = slice.binary_search(&x);
```

and we will get back the `index` in any case and continue on from there.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

[RFC 2005]: https://github.com/rust-lang/rfcs/blob/master/text/2005-match-ergonomics.md#examples

[RFC 2005], in describing the third example in the section "Examples", refers to
patterns with `|` in them as "or" patterns. This RFC adopts the same terminology.

While the "sum" of all patterns in `match` must be irrefutable, or in other
words: cover all cases, be exhaustive, this is not the case (currently) with
`if/while let`, which may have a refutable pattern.
This RFC does not change this.

The RFC only extends the use of or-patterns at the top level from `match`es
to `if let` and `while let` expressions as well as `let` statements.

For examples, see [motivation].

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Grammar

[§ 7.2.24]: https://doc.rust-lang.org/grammar.html#if-let-expressions
[§ 7.2.25]: https://doc.rust-lang.org/grammar.html#while-let-loops

### `if let`

The grammar in [§ 7.2.24] is changed from:

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

The grammar in [§ 7.2.25] is changed from:

```
while_let_expr : [ lifetime ':' ] ? "while" "let" pat '=' expr '{' block '}' ;
```

to:

```
while_let_expr : [ lifetime ':' ] ? "while" "let" pat [ '|' pat ] * '=' expr '{' block '}' ;
```

### `let` statements

The statement `stmt` grammar is replaced with a language equivalent to:

```
stmt ::= old_stmt_grammar
       | let_stmt_many
       ;

let_stmt_many ::= "let" pat_two_plus "=" expr ";"

pat_two_plus ::= pat [ '|' pat ] + ;
```

## Syntax lowering

The changes proposed in this RFC with respect to `if let` and `while let`
can be implemented by transforming the `if/while let` constructs with a
syntax-lowering pass into `match` and `loop` + `match` expressions.

Meanwhile, `let` statements can be transformed into a continuation with
`match` as described below.

### Examples, `if let`

[`if let` RFC]: https://github.com/rust-lang/rfcs/pull/160

These examples are extensions on the [`if let` RFC]. Therefore, the RFC avoids
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

[`while let` RFC]: https://github.com/rust-lang/rfcs/pull/214

The following example is an extension on the [`while let` RFC].

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

## Desugaring `let` statements with `|` in the top-level pattern

This is a possible desugaring that a Rust compiler may do.
While such a compiler may elect to implement this differently,
these semantics should be kept.

Source:
```rust
{
    // prefix of statements:
    stmt*
    // The let statement which is the cause for desugaring:
    let_stmt_many
    // the continuation / suffix of statements:
    stmt*
    tail_expr? // Meta-variable for optional tail expression without ; at end
}
```
Result
```rust
{
    stmt*
    match expr {
        pat_two_plus => {
            stmt*
            tail_expr?
        }
    }
}
```

# Drawbacks
[drawbacks]: #drawbacks

This adds more additions to the grammar and makes the compiler more complex.

# Rationale and alternatives
[alternatives]: #alternatives

This could simply not be done.
Consistency with `match` is however on its own reason enough to do this.

It could be claimed that the `if/while let` RFCs already mandate this RFC,
this RFC does answer that question and instead simply mandates it now.

Another alternative is to only deal with `if/while let` expressions but not
`let` statements.

# Unresolved questions
[unresolved]: #unresolved-questions

The exact syntax transformations should be deferred to the implementation.
This RFC does not mandate exactly how the AST:s should be transformed, only
that the or-pattern feature be supported.

There are no unresolved questions.
