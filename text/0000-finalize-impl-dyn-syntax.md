- Feature Name: N/A
- Start Date: 2017-12-16
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

"Alternative 3" (see reference-level explanation) is selected for stabilization.

`impl Trait1 + Trait2` / `dyn Trait1 + Trait2` now require parentheses in all
contexts - `impl(Trait1 + Trait2)` / `dyn(Trait1 + Trait2)`, similarly to other
unary operators, e.g. `&(Trait1 + Trait2)`.

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

In general, binary operations like `+` have lower priority than unary operations
in all contexts - expressions, patterns, types. So the priorities as implemented
bring inconsistency and may break intuition.

## Alternative 1: high priority `+` (status quo)

Pros:
- The greedy parsing with high priority of `+` after `impl` / `dyn`
has one benefit - it requires the least amout of parentheses from all the
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
fn f() -> impl A + B
{
    // Do things
}
```
This is probably the most common location for `impl Trait` types.
In theory, it doesn't require parentheses in any way - it's not inside of an
unary operator and it doesn't cross expression boundaries.  
However, it creates a bit of percieved inconsistency with function-like traits
and function pointers that do require parentheses for `impl Trait` in return
types (`Fn() -> (impl A + B)` / `fn() -> (impl A + B)`) because they, in their
turn, can appear inside of unary operators and casts.  
So, if avoiding this is considered more important than ergonomics, then
we can require parentheses in function definitions as well.
```rust
fn f() -> (impl A + B)
{
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
fn f() -> impl(A + B)
{
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

# Drawbacks
[drawbacks]: #drawbacks

See above.

# Rationale and alternatives
[alternatives]: #alternatives

See above.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
