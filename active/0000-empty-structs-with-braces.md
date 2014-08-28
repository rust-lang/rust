- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

When a struct type `S` has no fields (a so-called "empty struct"):

 * allow `S` to be defined via either `struct S;` (as today)
   or `struct S {}` (new)
 * allow instances of `S` to be constructed via either the
   expression `S` (as today) or the expression `S {}` (new)
 * allow instances of `S` to be pattern matched via either the
   pattern `S` (as today) or the pattern `S {}` (new).

# Motivation

Today, when writing code, one must treat an empty struct as a
special case, distinct from structs that include fields.
That is, one must write code like this:
```rust
struct S2 { x1: int, x2: int }
struct S0; // kind of different from the above.

let s2 = S2 { x1: 1, x2: 2 };
let s0 = S0; // kind of different from the above.

match (s2, s0) {
    (S2 { x1: y1, x2: y2 },
     S0) // you can see my pattern here
     => { println!("Hello from S2({}, {}) and S0", y1, y2); }
}
```

While this yields code that is relatively free of extraneous
curly-braces, this special case handling of empty structs presents
problems for two cases of interest: code generators (including, but
not limited to, Rust macros) and conditionalized code (i.e. code with
`cfg` attributes).

The special case handling of empty structs is also a problem for
programmers who actively add and remove fields from structs during
development; such changes cause a struct to switch from being empty
and non-empty, and the associated revisions of changing removing and
adding curly braces is aggravating (both in effort revising the code,
and also in extra noise introduced into commit histories).

This RFC proposes going back to the state we were in circa February
2013, when both `S0` and `S0 { }` were accepted syntaxes for an empty
struct.  The parsing ambiguity that motivated removing support for
`S0 { }` is no longer present (see [#ancient_history]).


# Detailed design

Revise the grammar of struct item definitions so that one can write
either `struct S;` or `struct S { }`.  The two forms are synonymous.
The first is preferred with respect to coding style; for example, the
first is emitted by the pretty printer.

Revise the grammar of expressions and patterns so that, when `S` is an
empty struct, one can write either `S` or `S { }`.  The two forms are
synonymous.  Again, the first is preferred with respect to coding style,
and is emitted by the pretty printer.

# Drawbacks

Some people like "There is only one way to do it."  But, there is
precendent in Rust for violating "one way to do it" in favor of
syntactic convenience or regularity; see
[#precedent_for_flexible_syntax_in_rust].
Also, see Alternative 1 below.

# Alternatives

Alternative 1: Require empty curly braces on empty structs.

Alternative 2: Status quo.  Macros and code-generators in general
will need to handle empty structs as a special case.  We may
continue hitting bugs like 

# Unresolved questions

None.

# Appendices

## Ancient History

A parsing ambiguity was the original motivation for disallowing the
syntax `struct S {}` in favor of `struct S;` for an empty struct
declaration.  The ambiguity and various options for dealing with it
were well documented on the [associated mailing list thread][RustDev
Thread].  Both syntaxes were simultaneously supported at the time.
Support for `struct S {}` was removed because that was the most
expedient option.  In particular, at that time, the option of "Place a
parser restriction on those contexts where `{` terminates the
expression and say that struct literals cannot appear there unless
they are in parentheses." was explicitly not chosen, in favor of
continuing to use the disambiguation rule in use at the time, namely
that the presence of a label (e.g. `S { a_label: ... }`) was *the* way
to distinguish a struct constructor from an identifier followed by a
control block, and thus, "there must be one label."

In particular, at the time that mailing list thread was created, the
code match `match x {} ...` would be parsed as `match (x {}) ...`, not
as `(match x {}) ...` (see [Rust PR 5137]); likewise, `if x {}` would
be parsed as an if-expression whose test component is the struct
literal `x {}`.  Thus, at the time of [Rust PR 5137], if the input to
a `match` or `if` was an identifier expression, one had to put
parentheses around the identifier to force it to be interpreted as
input, and not as a struct constructor.

Things have changed since then; namely, we have now adopted the
aforementioned parser restriction [Rust RFC 25].  (The text of RFC 25
does not explicitly address `match`, but we have effectively expanded
it to include a curly-brace delimited block of match-arms in the
definition of "block".)  Today, one uses parentheses around struct
literals in some contexts (such as `for e in (S {x: 3}) { ... }` or
`match (S {x: 3}) { ... }`

## Precedent for flexible syntax in Rust

There is precendent in Rust for violating "one way to do it" in favor
of syntactic convenience or regularity.

For example, one can often include an optional trailing comma, for
example in: `let x : &[int] = [3, 2, 1, ];`.

One can also include redundant curly braces or parentheses, for
example in:
```rust
println!("hi: {}", { if { x.len() > 2 } { ("whoa") } else { ("there") } });
```

One can even mix the two together when delimiting match arms:
```rust
    let z: int = match x {
        [3, 2] => { 3 }
        [3, 2, 1] => 2,
        _ => { 1 },
    };
```

We do have lints for some style violations (though none catch the
cases above), but lints are different from fundamental language
restrictions.


[RustDev Thread]: https://mail.mozilla.org/pipermail/rust-dev/2013-February/003282.html

[Rust Issue 5167]: https://github.com/rust-lang/rust/issues/5167

[Rust RFC 25]: https://github.com/rust-lang/rfcs/blob/master/complete/0025-struct-grammar.md

[CFG parse bug]: https://github.com/rust-lang/rust/issues/16819

[Rust PR 5137]: https://github.com/rust-lang/rust/pull/5137
