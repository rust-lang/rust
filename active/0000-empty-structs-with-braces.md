- Start Date: (fill me in with today's date, 2014-08-28)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

When a struct type `S` has no fields (a so-called "empty struct"),
allow it to be defined via either `struct S;` or `struct S {}`, and
allow instances of it to be constructed and pattern-matched via either
`S` or `S {}`.

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
problems for two cases of interest: automatic code generators
(including, but not limited to, Rust macros) and conditionalized code
(i.e. code with `cfg` attributes; see appendix [#the_cfg_problem]).
The heart of the code-generator argument is: Why force all
to-be-written code-generators and macros with special-case handling of
the empty struct case (in terms of whether or not to include the
surrounding braces), especially since that special case is likely to
be forgotten (yielding a latent bug in the code generator).

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
Supporting empty braces in the syntax for empty structs is easy to do
in the language now.

# Detailed design

 * Allow `S` to be defined via either `struct S;` (as today)
   or `struct S {}` (new)

 * Allow instances of `S` to be constructed via either the
   expression `S` (as today) or the expression `S {}` (new)

 * Allow instances of `S` to be pattern matched via either the
   pattern `S` (as today) or the pattern `S {}` (new).

Revise the grammar of struct item definitions so that one can write
either `struct S;` or `struct S { }`.  The two forms are synonymous.
The first is preferred with respect to coding style; for example, the
first is emitted by the pretty printer.

Revise the grammar of expressions and patterns so that, when `S` is an
empty struct, one can write either `S` or `S { }`.  The two forms are
synonymous.  Again, the first is preferred with respect to coding
style, and is emitted by the pretty printer.

The format of the definiton has no bearing on the format of the
expressions or pattern forms; either syntax can be used for any
empty-struct, regardless of how it is defined.

There is no ambiguity introduced by this change, because we have
already introduced a restriction to the Rust grammar to force the use
of parentheses to disambiguate struct literals in such contexts.  (See
[Rust RFC 25]).


# Drawbacks

Some people like "There is only one way to do it."  But, there is
precendent in Rust for violating "one way to do it" in favor of
syntactic convenience or regularity; see
the appendix
[Precedent for flexible syntax in Rust][#precedent_for_flexible_syntax_in_rust].
Also, see Alternative 1: "Always Require Braces" below.

# Alternatives

Alternative 1: "Always Require Braces".  Specifically, require empty
curly braces on empty structs.  People who like the current syntax of
curly-brace free structs can encode them this way: `enum S0 { S0 }`
This would address all of the same issues outlined above. (Also, the
author (pnkfelix) would be happy to take this tack.)  The main reason
not to take this tack is that some people may like writing empty
structs without braces, but do not want to switch to the unary enum
version.  See "I wouldn't want to force noisier syntax ..." in
[#recent_history].

Alternative 2: Status quo.  Macros and code-generators in general
will need to handle empty structs as a special case.  We may
continue hitting bugs like 

# Unresolved questions

## Empty Tuple Structs

The code-generation argument could be applied to tuple-structs as
well, to claim that we should allow the syntax `S0()`.  I am less
inclined to add a special case for that.  Note that we should not
attempt to generalize this RFC as proposed to include tuple structs,
i.e. so that given `struct S0 {}`, the expressions `T0`, `T0 {}`, and
`T0()` would be synonymous.  The reason is that
given a tuple struct `struct T2(int, int)`, the identifier `T2` is
*already* bound to the constructor function:

```rust
fn main() {
    #[deriving(Show)]
    struct T2(int, int);

    fn foo<S:std::fmt::Show>(f: |int, int| -> S) {
        println!("Hello from {} and {}", f(2,3), f(4,5));
    }
    foo(T2);
}
```

So if we were to attempt to generalize the leniency of this RFC to
tuple structs, we would be in the unfortunate situation given `struct
T0();` of trying to treat `T0` simultaneously as an instance of the
struct and as a constructor function.  So, the handling of empty
structs proposed by this RFC does not generalize to tuple structs.

(Note that if we adopt alternative 1, then the issue of how tuple
structs are handled is totally orthogonal -- we could add support for
`struct T0()` as a distinct type from `struct S0 {}`, if we so wished,
or leave it aside.)

# Appendices

## The CFG problem

A program like this works today:

```rust
fn main() {
    #[deriving(Show)]
    struct Svaries {
        x: int,
        y: int,

        #[cfg(zed)]
        z: int,
    }

    let s = match () {
        #[cfg(zed)]      _ => Svaries { x: 3, y: 4, z: 5 },
        #[cfg(not(zed))] _ => Svaries { x: 3, y: 4 },
    };
    println!("Hello from {}", s)
}
```

Observe what happens when one modifies the above just a bit:
```rust
    struct Svaries {
        #[cfg(eks)]
        x: int,
        #[cfg(why)]
        y: int,

        #[cfg(zed)]
        z: int,
    }
```

Now, certain `cfg` settings yield an empty struct, even though it
is surrounded by braces.  Today this leads to a [CFG parse bug].

If we want to support situations like this properly, we will probably
need to further extend the `cfg` attribute so that it can be placed
before individual fields in a struct constructor, like this:

```rust
// You cannot do this today,
// but maybe in the future (after a different RFC)
let s = Svaries {
    #[cfg(eks)] x: 3,
    #[cfg(why)] y: 4,
    #[cfg(zed)] z: 5,
};
```

Supporting such a syntax consistently in the future should start today
with allowing empty braces as legal code.  (Strictly speaking, it is
not *necessary* that we add support for empty braces at the parsing
level to support this feature at the semantic level.  But supporting
empty-braces in the syntax still seems like the most consistent path
to me.)

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

Note that there was never an ambiguity for uses of `struct S0 { }` in item
position.  The issue was solely about expression position prior to the
adoption of [Rust RFC 25].

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

## Recent history

There was a previous [RFC PR][RFC PR 147] that was effectively the
same in spirit to this one.  It was closed because it was not
sufficient well fleshed out for further consideration by the core
team.  However, to save people the effort of reviewing the comments on
that PR (and hopefully stave off potential bikeshedding on this PR), I
here summarize the various viewpoints put forward on the comment
thread there, and note for each one, whether that viewpoint would be
addressed by this RFC (accept both syntaxes), by Alternative 1 (accept
only `S0 {}`), or by the status quo (accept only `S0`).



* "I find `let s = S0;` jarring, think its an enum initially." ==> Favors: Alternative 1
* "Frequently start out with an empty struct and add fields as I need them." ==> Favors: This RFC or Alternative 1
* "Foo{} suggests is constructing something that it's not; all uses of the value `Foo` are indistinguishable from each other" ==> Favors: Status Quo
* "I find it strange anyone would prefer `let x = Foo{};` over `let x = Foo;`" ==> Favors Status Quo; strongly opposes Alternative 1.
* "I agree that 'instantiation-should-follow-declation', that is, structs declared `;, (), {}` should only be instantiated [via] `;, (), { }` respectively" ==> Opposes leniency of this RFC in that it allows expression to use include or omit `{}` on an empty struct, regardless of declaration form, and vice-versa.
* "The code generation argument is reasonable, but I wouldn't want to force noisier syntax on all 'normal' code just to make macros work better." ==> Favors: This RFC

[RustDev Thread]: https://mail.mozilla.org/pipermail/rust-dev/2013-February/003282.html

[Rust Issue 5167]: https://github.com/rust-lang/rust/issues/5167

[Rust RFC 25]: https://github.com/rust-lang/rfcs/blob/master/complete/0025-struct-grammar.md

[CFG parse bug]: https://github.com/rust-lang/rust/issues/16819

[Rust PR 5137]: https://github.com/rust-lang/rust/pull/5137

[RFC PR 147]: https://github.com/rust-lang/rfcs/pull/147
