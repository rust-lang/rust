- Start Date: 2014-08-26
- RFC PR #: [rust-lang/rfcs#160](https://github.com/rust-lang/rfcs/pull/160)
- Rust Issue #: [rust-lang/rust#16779](https://github.com/rust-lang/rust/issues/16779)

# Summary

Introduce a new `if let PAT = EXPR { BODY }` construct. This allows for refutable pattern matching
without the syntactic and semantic overhead of a full `match`, and without the corresponding extra
rightward drift. Informally this is known as an "if-let statement".

# Motivation

Many times in the past, people have proposed various mechanisms for doing a refutable let-binding.
None of them went anywhere, largely because the syntax wasn't great, or because the suggestion
introduced runtime failure if the pattern match failed.

This proposal ties the refutable pattern match to the pre-existing conditional construct (i.e. `if`
statement), which provides a clear and intuitive explanation for why refutable patterns are allowed
here (as opposed to a `let` statement which disallows them) and how to behave if the pattern doesn't
match.

The motivation for having any construct at all for this is to simplify the cases that today call for
a `match` statement with a single non-trivial case. This is predominately used for unwrapping
`Option<T>` values, but can be used elsewhere.

The idiomatic solution today for testing and unwrapping an `Option<T>` looks like

```rust
match optVal {
    Some(x) => {
        doSomethingWith(x);
    }
    None => {}
}
```

This is unnecessarily verbose, with the `None => {}` (or `_ => {}`) case being required, and
introduces unnecessary rightward drift (this introduces two levels of indentation where a normal
conditional would introduce one).

The alternative approach looks like this:

```rust
if optVal.is_some() {
    let x = optVal.unwrap();
    doSomethingWith(x);
}
```

This is generally considered to be a less idiomatic solution than the `match`. It has the benefit of
fixing rightward drift, but it ends up testing the value twice (which should be optimized away, but
semantically speaking still happens), with the second test being a method that potentially
introduces failure. From context, the failure won't happen, but it still imposes a semantic burden
on the reader. Finally, it requires having a pre-existing let-binding for the optional value; if the
value is a temporary, then a new let-binding in the parent scope is required in order to be able to
test and unwrap in two separate expressions.

The `if let` construct solves all of these problems, and looks like this:

```rust
if let Some(x) = optVal {
    doSomethingWith(x);
}
```

# Detailed design

The `if let` construct is based on the precedent set by Swift, which introduced its own `if let`
statement. In Swift, `if let var = expr { ... }` is directly tied to the notion of optional values,
and unwraps the optional value that `expr` evaluates to. In this proposal, the equivalent is `if let
Some(var) = expr { ... }`.

Given the following rough grammar for an `if` condition:

```
if-expr     = 'if' if-cond block else-clause?
if-cond     = expression
else-clause = 'else' block | 'else' if-expr
```

The grammar is modified to add the following productions:

```
if-cond = 'let' pattern '=' expression
```

The `expression` is restricted to disallow a trailing braced block (e.g. for struct literals) the
same way the `expression` in the normal `if` statement is, to avoid ambiguity with the then-block.

Contrary to a `let` statement, the pattern in the `if let` expression allows refutable patterns. The
compiler should emit a warning for an `if let` expression with an irrefutable pattern, with the
suggestion that this should be turned into a regular `let` statement.

Like the `for` loop before it, this construct can be transformed in a syntax-lowering pass into the
equivalent `match` statement. The `expression` is given to `match` and the `pattern` becomes a match
arm. If there is an `else` block, that becomes the body of the `_ => {}` arm, otherwise `_ => {}` is
provided.

Optionally, one or more `else if` (not `else if let`) blocks can be placed in the same `match` using
pattern guards on `_`. This could be done to simplify the code when pretty-printing the expansion
result. Otherwise, this is an unnecessary transformation.

Due to some uncertainty regarding potentially-surprising fallout of AST rewrites, and some worries
about exhaustiveness-checking (e.g. a tautological `if let` would be an error, which may be
unexpected), this is put behind a feature gate named `if_let`.

## Examples

Source:

```rust
if let Some(x) = foo() {
    doSomethingWith(x)
}
```

Result:

```rust
match foo() {
    Some(x) => {
        doSomethingWith(x)
    }
    _ => {}
}
```

Source:

```rust
if let Some(x) = foo() {
    doSomethingWith(x)
} else {
    defaultBehavior()
}
```

Result:

```rust
match foo() {
    Some(x) => {
        doSomethingWith(x)
    }
    _ => {
        defaultBehavior()
    }
}
```

Source:

```rust
if cond() {
    doSomething()
} else if let Some(x) = foo() {
    doSomethingWith(x)
} else {
    defaultBehavior()
}
```

Result:

```rust
if cond() {
    doSomething()
} else {
    match foo() {
        Some(x) => {
            doSomethingWith(x)
        }
        _ => {
            defaultBehavior()
        }
    }
}
```

With the optional addition specified above:

```rust
if let Some(x) = foo() {
    doSomethingWith(x)
} else if cond() {
    doSomething()
} else if other_cond() {
    doSomethingElse()
}
```

Result:

```rust
match foo() {
    Some(x) => {
        doSomethingWith(x)
    }
    _ if cond() => {
        doSomething()
    }
    _ if other_cond() => {
        doSomethingElse()
    }
    _ => {}
}
```

# Drawbacks

It's one more addition to the grammar.

# Alternatives

This could plausibly be done with a macro, but the invoking syntax would be pretty terrible and
would largely negate the whole point of having this sugar.

Alternatively, this could not be done at all. We've been getting alone just fine without it so far,
but at the cost of making `Option` just a bit more annoying to work with.

# Unresolved questions

It's been suggested that alternates or pattern guards should be allowed. I think if you need those
you could just go ahead and use a `match`, and that `if let` could be extended to support those in
the future if a compelling use-case is found.

I don't know how many `match` statements in our current code base could be replaced with this
syntax. Probably quite a few, but it would be informative to have real data on this.
