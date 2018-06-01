- Feature Name: N/A
- Start Date: 2017-12-16
- RFC PR: [rust-lang/rfcs#2250](https://github.com/rust-lang/rfcs/pull/2250)
- Rust Issue: [rust-lang/rust#34511](https://github.com/rust-lang/rust/issues/34511)

# Summary
[summary]: #summary

Finalize syntax of `impl Trait` and `dyn Trait` with multiple bounds before
stabilization of these features.

# Motivation
[motivation]: #motivation

Current priority of `+` in `impl Trait1 + Trait2` / `dyn Trait1 + Trait2` brings
inconsistency in the type grammar.
This RFC outlines possible syntactic
alternatives and suggests one of them for stabilization.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

"Alternative 2" (see reference-level explanation) is selected for stabilization.

`impl Trait1 + Trait2` / `dyn Trait1 + Trait2` now require parentheses in all
contexts where they are used inside of unary operators `&(impl Trait1 + Trait2)`
/ `&(dyn Trait1 + Trait2)`, similarly to trait object types without
prefix, e.g. `&(Trait1 + Trait2)`.

Additionally, parentheses are required in all cases where `+` in `impl` or `dyn`
is ambiguous.
For example, `Fn() -> impl A + B` can be interpreted as both
`(Fn() -> impl A) + B` (low priority plus) or `Fn() -> (impl A + B)` (high
priority plus), so we are refusing to disambiguate and require explicit
parentheses.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Current situation

In the current implementation when we see `impl` or `dyn` we start parsing
following bounds separated by `+`s greedily regardless of context, so `+`
effectively gets the strongest priority.

So, for example:
- `&dyn A + B` is parsed as `&(dyn A + B)`
- `Fn() -> impl A + B` is parsed as `Fn() -> (impl A + B)`
- `x as &dyn A + y` is parsed as `x as &(dyn A + y)`.

Compare this with parsing of trait object types without prefixes
([RFC 438](https://github.com/rust-lang/rfcs/pull/438)):
- `&A + B` is parsed as `(&A) + B` and is an error
- `Fn() -> A + B` is parsed as `(Fn() -> A) + B`
- `x as &A + y` is parsed as `(x as &A) + y`

Also compare with unary operators in bounds themselves:
- `for<'a> A<'a> + B` is parsed as `(for<'a> A<'a>) + B`,
not `for<'a> (A<'a> + B)`
- `?A + B` is parsed as `(?A) + B`, not `?(A + B)`

In general, binary operations like `+` have lower priority than unary operations
in all contexts - expressions, patterns, types. So the priorities as implemented
bring inconsistency and may break intuition.

## Alternative 1: high priority `+` (status quo)

Pros:
- The greedy parsing with high priority of `+` after `impl` / `dyn`
has one benefit - it requires the least amount of parentheses from all the
alternatives.
Parentheses are needed only when the greedy behaviour needs to be prevented,
e.g. `Fn() -> &(dyn Write) + Send`, this doesn't happen often.

Cons:
- Inconsistent and possibly surprising operator priorities.
- `impl` / `dyn` is a somewhat weird syntactic construction, it's not an usual
unary operator, its a prefix describing how to interpret the following tokens.
In particular, if the `impl A + B` needs to be parenthesized for some reason,
it needs to be done like this `(impl A + B)`, and not `impl (A + B)`. The second
variant is a parsing error, but some people find it surprising and expect it to
work, as if `impl` were an unary operator.

## Alternative 2: low priority `+`

Basically, `impl A + B` is parsed using same rules as `A + B`.

If `impl A + B` is located inside a higher priority operator like `&` it has
to be parenthesized.
If it is located at intersection of type and expressions
grammars like `expr1 as Type + expr2`, it has to be parenthesized as well.

`&dyn A + B` / `Fn() -> impl A + B` / `x as &dyn A + y` has to be rewritten as
`&(dyn A + B)` / `Fn() -> (impl A + B)` / `x as &(dyn A + y)` respectively.

One location must be mentioned specially, the location in a function return
type:
```rust
fn f() -> impl A + B {
    // Do things
}
```
This is probably the most common location for `impl Trait` types.
In theory, it doesn't require parentheses in any way - it's not inside of an
unary operator and it doesn't cross expression boundaries.
However, it creates a bit of perceived inconsistency with function-like traits
and function pointers that do require parentheses for `impl Trait` in return
types (`Fn() -> (impl A + B)` / `fn() -> (impl A + B)`) because they, in their
turn, can appear inside of unary operators and casts.
So, if avoiding this is considered more important than ergonomics, then
we can require parentheses in function definitions as well.
```rust
fn f() -> (impl A + B) {
    // Do things
}
```

Pros:
- Consistent priorities of binary and unary operators.
- Parentheses are required relatively rarely (unless we require them in
function definitions as well).

Cons:
- More parentheses than in the "Alternative 1".
- `impl` / `dyn` is still a somewhat weird prefix construction and `dyn (A + B)`
is not a valid syntax.

## Alternative 3: Unary operator

`impl` and `dyn` can become usual unary operators in type grammar like `&` or
`*const`.
Their application to any other types except for (possibly parenthesized) paths
(single `A`) or "legacy trait objects" (`A + B`) becomes an error, but this
could be changed in the future if some other use is found.

`&dyn A + B` / `Fn() -> impl A + B` / `x as &dyn A + y` has to be rewritten as
`&dyn(A + B)` / `Fn() -> impl(A + B)` / `x as &dyn(A + y)` respectively.

Function definitions with `impl A + B` in return type have to be rewritten too.
```rust
fn f() -> impl(A + B) {
    // Do things
}
```

Pros:
- Consistent priorities of binary and unary operators.
- `impl` / `dyn` are usual unary operators, `dyn (A + B)` is a valid syntax.

Cons:
- The largest amount of parentheses, parentheses are always required.
Parentheses are noise, there may be even less desire to use `dyn` in trait
objects now, if something like `Box<Write + Send>` turns into
`Box<dyn(Write + Send)>`.

## Other alternatives

Two separate grammars can be used depending on context
(https://github.com/rust-lang/rfcs/pull/2250#issuecomment-352435687) -
Alternative 1/2 in lists of arguments like `Box<dyn A + B>` or
`Fn(impl A + B, impl A + B)`, and Alternative 3 otherwise (`&dyn (A + B)`).

## Compatibility

The alternatives are ordered by strictness from the most relaxed Alternative 1
to the strictest Alternative 3, but switching from more strict alternatives to
less strict is not exactly backward-compatible.

Switching from 2/3 to 1 can change meaning of legal code in rare cases.
Switching from 3 to 2/1 requires keeping around the syntax with parentheses
after `impl` / `dyn`.

Alternative 2 can be backward-compatibly extended to "relaxed 3" in which
parentheses like `dyn (A + B)` are permitted, but technically unnecessary.
Such parenthesis may keep people expecting `dyn (A + B)` to work happy, but
complicate parsing by introducing more ambiguities to the grammar.

While unary operators like `&` "obviously" have higher priority than `+`,
cases like `Fn() -> impl A + B` are not so obvious.
The Alternative 2 considers "low priority plus" to have lower priority than `Fn`
, so `Fn() -> impl A + B` can be treated as `(Fn() -> impl A) + B`, however
it may be more intuitive and consistent with `fn` items to make `+` have higher
priority than `Fn` (but still lower priority than `&`).
As an immediate solution we refuse to disambiguate this case and treat
`Fn() -> impl A + B` as an error, so we can change the rules in the future and
interpret `Fn() -> impl A + B` (and maybe even `Fn() -> A + B` after long
deprecation period) as `Fn() -> (impl A + B)` (and `Fn() -> (A + B)`,
respectively).

## Experimental check

An application of all the alternatives to rustc and libstd codebase can be found
in [this branch](https://github.com/petrochenkov/rust/commits/impldyntest).
The first commit is the baseline (Alternative 1) and the next commits show
changes required to move to Alternatives 2 and 3. Alternative 2 requires fewer
changes compared to Alternative 3.

As the RFC author interprets it, the Alternative 3 turns out to be impractical
due to common use of `Box`es and other contexts where the parenthesis are technically
unnecessary, but required by Alternative 3.
The number of parenthesis required by Alternative 2 is limited and they seem
appropriate because they follow "normal" priorities for unary and binary
operators.

# Drawbacks
[drawbacks]: #drawbacks

See above.

# Rationale and alternatives
[alternatives]: #alternatives

See above.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
