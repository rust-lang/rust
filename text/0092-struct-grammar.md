- Start Date: 2014-06-10
- RFC PR: [rust-lang/rfcs#92](https://github.com/rust-lang/rfcs/pull/92)
- Rust Issue: [rust-lang/rust#14803](https://github.com/rust-lang/rust/issues/14803)

# Summary

Do not identify struct literals by searching for `:`. Instead define a sub-
category of expressions which excludes struct literals and re-define `for`,
`if`, and other expressions which take an expression followed by a block (or
non-terminal which can be replaced by a block) to take this sub-category,
instead of all expressions.

# Motivation

Parsing by looking ahead is fragile - it could easily be broken if we allow `:`
to appear elsewhere in types (e.g., type ascription) or if we change struct
literals to not require the `:` (e.g., if we allow empty structs to be written
with braces, or if we allow struct literals to unify field names to local
variable names, as has been suggested in the past and which we currently do for
struct literal patterns). We should also be able to give better error messages
today if users make these mistakes. More worringly, we might come up with some
language feature in the future which is not predictable now and which breaks
with the current system.

Hopefully, it is pretty rare to use struct literals in these positions, so there
should not be much fallout. Any problems can be easily fixed by assigning the
struct literal into a variable. However, this is a backwards incompatible
change, so it should block 1.0.

# Detailed design

Here is a simplified version of a subset of Rust's abstract syntax:

```
e      ::= x
         | e `.` f
         | name `{` (x `:` e)+ `}`
         | block
         | `for` e `in` e block
         | `if` e block (`else` block)?
         | `|` pattern* `|` e
         | ...
block  ::=  `{` (e;)* e? `}`
```

Parsing this grammar is ambiguous since `x` cannot be distinguished from `name`,
so `e block` in the for expression is ambiguous with the struct literal
expression. We currently solve this by using lookahead to find a `:` token in
the struct literal.

I propose the following adjustment:

```
e      ::= e'
         | name `{` (x `:` e)+ `}`
         | `|` pattern* `|` e
         | ...
e'     ::= x
         | e `.` f
         | block
         | `for` e `in` e' block
         | `if` e' block (`else` block)?
         | `|` pattern* `|` e'
         | ...
block  ::=  `{` (e;)* e? `}`
```

`e' is just e without struct literal expressions. We use e' instead of e
`wherever e is followed directly by block or any other non-terminal which may
`have block as its first terminal (after any possible expansions).

For any expressions where a sub-expression is the final lexical element
(closures in the subset above, but also unary and binary operations), we require
two versions of the meta-expression - the normal one in `e` and a version with
`e'` for the final element in `e'`.

Implementation would be simpler, we just add a flag to `parser::restriction`
called `RESTRICT_BLOCK` or something, which puts us into a mode which reflects
`e'`. We would drop in to this mode when parsing `e'` position expressions and
drop out of it for all but the last sub-expression of an expression.

# Drawbacks

It makes the formal grammar and parsing a little more complicated (although it
is simpler in terms of needing less lookahead and avoiding a special case).

# Alternatives

Don't do this.

Allow all expressions but greedily parse non-terminals in these positions, e.g.,
`for N {} {}` would be parsed as `for (N {}) {}`. This seems worse because I
believe it will be much rarer to have structs in these positions than to have an
identifier in the first position, followed by two blocks (i.e., parse as `(for N
{}) {}`).

# Unresolved questions

Do we need to expose this distinction anywhere outside of the parser? E.g.,
macros?
