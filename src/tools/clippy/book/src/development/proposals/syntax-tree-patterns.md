- Feature Name: syntax-tree-patterns
- Start Date: 2019-03-12
- RFC PR: [#3875](https://github.com/rust-lang/rust-clippy/pull/3875)

# Summary

Introduce a domain-specific language (similar to regular expressions) that
allows to describe lints using *syntax tree patterns*.


# Motivation

Finding parts of a syntax tree (AST, HIR, ...) that have certain properties
(e.g. "*an if that has a block as its condition*") is a major task when writing
lints. For non-trivial lints, it often requires nested pattern matching of AST /
HIR nodes. For example, testing that an expression is a boolean literal requires
the following checks:

```rust
if let ast::ExprKind::Lit(lit) = &expr.node {
    if let ast::LitKind::Bool(_) = &lit.node {
        ...
    }
}
```

Writing this kind of matching code quickly becomes a complex task and the
resulting code is often hard to comprehend. The code below shows a simplified
version of the pattern matching required by the `collapsible_if` lint:

```rust
// simplified version of the collapsible_if lint
if let ast::ExprKind::If(check, then, None) = &expr.node {
    if then.stmts.len() == 1 {
        if let ast::StmtKind::Expr(inner) | ast::StmtKind::Semi(inner) = &then.stmts[0].node {
            if let ast::ExprKind::If(check_inner, content, None) = &inner.node {
                ...
            }
        }
    }
}
```

The `if_chain` macro can improve readability by flattening the nested if
statements, but the resulting code is still quite hard to read:

```rust
// simplified version of the collapsible_if lint
if_chain! {
    if let ast::ExprKind::If(check, then, None) = &expr.node;
    if then.stmts.len() == 1;
    if let ast::StmtKind::Expr(inner) | ast::StmtKind::Semi(inner) = &then.stmts[0].node;
    if let ast::ExprKind::If(check_inner, content, None) = &inner.node;
    then {
        ...
    }
}
```

The code above matches if expressions that contain only another if expression
(where both ifs don't have an else branch). While it's easy to explain what the
lint does, it's hard to see that from looking at the code samples above.

Following the motivation above, the first goal this RFC is to **simplify writing
and reading lints**.

The second part of the motivation is clippy's dependence on unstable
compiler-internal data structures. Clippy lints are currently written against
the compiler's AST / HIR which means that even small changes in these data
structures might break a lot of lints. The second goal of this RFC is to **make
lints independent of the compiler's AST / HIR data structures**.

# Approach

A lot of complexity in writing lints currently seems to come from having to
manually implement the matching logic (see code samples above). It's an
imperative style that describes *how* to match a syntax tree node instead of
specifying *what* should be matched against declaratively. In other areas, it's
common to use declarative patterns to describe desired information and let the
implementation do the actual matching. A well-known example of this approach are
[regular expressions](https://en.wikipedia.org/wiki/Regular_expression). Instead
of writing code that detects certain character sequences, one can describe a
search pattern using a domain-specific language and search for matches using
that pattern. The advantage of using a declarative domain-specific language is
that its limited domain (e.g. matching character sequences in the case of
regular expressions) allows to express entities in that domain in a very natural
and expressive way.

While regular expressions are very useful when searching for patterns in flat
character sequences, they cannot easily be applied to hierarchical data
structures like syntax trees. This RFC therefore proposes a pattern matching
system that is inspired by regular expressions and designed for hierarchical
syntax trees.

# Guide-level explanation

This proposal adds a `pattern!` macro that can be used to specify a syntax tree
pattern to search for. A simple pattern is shown below:

```rust
pattern!{
    my_pattern: Expr =
        Lit(Bool(false))
}
```

This macro call defines a pattern named `my_pattern` that can be matched against
an `Expr` syntax tree node. The actual pattern (`Lit(Bool(false))` in this case)
defines which syntax trees should match the pattern. This pattern matches
expressions that are boolean literals with value `false`.

The pattern can then be used to implement lints in the following way:

```rust
...

impl EarlyLintPass for MyAwesomeLint {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &syntax::ast::Expr) {

        if my_pattern(expr).is_some() {
            cx.span_lint(
                MY_AWESOME_LINT,
                expr.span,
                "This is a match for a simple pattern. Well done!",
            );
        }

    }
}
```

The `pattern!` macro call expands to a function `my_pattern` that expects a
syntax tree expression as its argument and returns an `Option` that indicates
whether the pattern matched.

> Note: The result type is explained in more detail in [a later
> section](#the-result-type). For now, it's enough to know that the result is
> `Some` if the pattern matched and `None` otherwise.

## Pattern syntax

The following examples demonstate the pattern syntax:


#### Any (`_`)

The simplest pattern is the any pattern. It matches anything and is therefore
similar to regex's `*`.

```rust
pattern!{
    // matches any expression
    my_pattern: Expr =
        _
}
```

#### Node (`<node-name>(<args>)`)

Nodes are used to match a specific variant of an AST node. A node has a name and
a number of arguments that depends on the node type. For example, the `Lit` node
has a single argument that describes the type of the literal. As another
example, the `If` node has three arguments describing the if's condition, then
block and else block.

```rust
pattern!{
    // matches any expression that is a literal
    my_pattern: Expr =
        Lit(_)
}

pattern!{
    // matches any expression that is a boolean literal
    my_pattern: Expr =
        Lit(Bool(_))
}

pattern!{
    // matches if expressions that have a boolean literal in their condition
    // Note: The `_?` syntax here means that the else branch is optional and can be anything.
    //       This is discussed in more detail in the section `Repetition`.
    my_pattern: Expr =
        If( Lit(Bool(_)) , _, _?)
}
```


#### Literal (`<lit>`)

A pattern can also contain Rust literals. These literals match themselves.

```rust
pattern!{
    // matches the boolean literal false
    my_pattern: Expr =
        Lit(Bool(false))
}

pattern!{
    // matches the character literal 'x'
    my_pattern: Expr =
        Lit(Char('x'))
}
```

#### Alternations (`a | b`)

```rust
pattern!{
    // matches if the literal is a boolean or integer literal
    my_pattern: Lit =
        Bool(_) | Int(_)
}

pattern!{
    // matches if the expression is a char literal with value 'x' or 'y'
    my_pattern: Expr =
        Lit( Char('x' | 'y') )
}
```

#### Empty (`()`)

The empty pattern represents an empty sequence or the `None` variant of an
optional.

```rust
pattern!{
    // matches if the expression is an empty array
    my_pattern: Expr =
        Array( () )
}

pattern!{
    // matches if expressions that don't have an else clause
    my_pattern: Expr =
        If(_, _, ())
}
```

#### Sequence (`<a> <b>`)

```rust
pattern!{
    // matches the array [true, false]
    my_pattern: Expr =
        Array( Lit(Bool(true)) Lit(Bool(false)) )
}
```

#### Repetition (`<a>*`, `<a>+`, `<a>?`, `<a>{n}`, `<a>{n,m}`, `<a>{n,}`)

Elements may be repeated. The syntax for specifying repetitions is identical to
[regex's syntax](https://docs.rs/regex/1.1.2/regex/#repetitions).

```rust
pattern!{
    // matches arrays that contain 2 'x's as their last or second-last elements
    // Examples:
    //     ['x', 'x']                         match
    //     ['x', 'x', 'y']                    match
    //     ['a', 'b', 'c', 'x', 'x', 'y']     match
    //     ['x', 'x', 'y', 'z']               no match
    my_pattern: Expr =
        Array( _* Lit(Char('x')){2} _? )
}

pattern!{
    // matches if expressions that **may or may not** have an else block
    // Attn: `If(_, _, _)` matches only ifs that **have** an else block
    //
    //              | if with else block | if without else block
    // If(_, _, _)  |       match        |       no match
    // If(_, _, _?) |       match        |        match
    // If(_, _, ()) |      no match      |        match
    my_pattern: Expr =
        If(_, _, _?)
}
```

#### Named submatch (`<a>#<name>`)

```rust
pattern!{
    // matches character literals and gives the literal the name foo
    my_pattern: Expr =
        Lit(Char(_)#foo)
}

pattern!{
    // matches character literals and gives the char the name bar
    my_pattern: Expr =
        Lit(Char(_#bar))
}

pattern!{
    // matches character literals and gives the expression the name baz
    my_pattern: Expr =
        Lit(Char(_))#baz
}
```

The reason for using named submatches is described in the section [The result
type](#the-result-type).

### Summary

The following table gives an summary of the pattern syntax:

| Syntax                  | Concept          | Examples                                   |
|-------------------------|------------------|--------------------------------------------|
|`_`                      | Any              | `_`                                        |
|`<node-name>(<args>)`    | Node             | `Lit(Bool(true))`, `If(_, _, _)`           |
|`<lit>`                  | Literal          | `'x'`, `false`, `101`                      |
|`<a> \| <b>`             | Alternation      | `Char(_) \| Bool(_)`                       |
|`()`                     | Empty            | `Array( () )`                              |
|`<a> <b>`                | Sequence         | `Tuple( Lit(Bool(_)) Lit(Int(_)) Lit(_) )` |
|`<a>*` <br> `<a>+` <br> `<a>?` <br> `<a>{n}` <br> `<a>{n,m}` <br> `<a>{n,}` | Repetition <br> <br> <br> <br> <br><br> | `Array( _* )`, <br> `Block( Semi(_)+ )`, <br> `If(_, _, Block(_)?)`, <br> `Array( Lit(_){10} )`, <br> `Lit(_){5,10}`, <br> `Lit(Bool(_)){10,}` |
|`<a>#<name>`             | Named submatch   | `Lit(Int(_))#foo` `Lit(Int(_#bar))`        |


## The result type

A lot of lints require checks that go beyond what the pattern syntax described
above can express. For example, a lint might want to check whether a node was
created as part of a macro expansion or whether there's no comment above a node.
Another example would be a lint that wants to match two nodes that have the same
value (as needed by lints like `almost_swapped`). Instead of allowing users to
write these checks into the pattern directly (which might make patterns hard to
read), the proposed solution allows users to assign names to parts of a pattern
expression. When matching a pattern against a syntax tree node, the return value
will contain references to all nodes that were matched by these named
subpatterns. This is similar to capture groups in regular expressions.

For example, given the following pattern

```rust
pattern!{
    // matches character literals
    my_pattern: Expr =
        Lit(Char(_#val_inner)#val)#val_outer
}
```

one could get references to the nodes that matched the subpatterns in the
following way:

```rust
...
fn check_expr(expr: &syntax::ast::Expr) {
    if let Some(result) = my_pattern(expr) {
        result.val_inner  // type: &char
        result.val        // type: &syntax::ast::Lit
        result.val_outer  // type: &syntax::ast::Expr
    }
}
```

The types in the `result` struct depend on the pattern. For example, the
following pattern

```rust
pattern!{
    // matches arrays of character literals
    my_pattern_seq: Expr =
        Array( Lit(_)*#foo )
}
```

matches arrays that consist of any number of literal expressions. Because those
expressions are named `foo`, the result struct contains a `foo` attribute which
is a vector of expressions:

```rust
...
if let Some(result) = my_pattern_seq(expr) {
    result.foo        // type: Vec<&syntax::ast::Expr>
}
```

Another result type occurs when a name is only defined in one branch of an
alternation:

```rust
pattern!{
    // matches if expression is a boolean or integer literal
    my_pattern_alt: Expr =
        Lit( Bool(_#bar) | Int(_) )
}
```

In the pattern above, the `bar` name is only defined if the pattern matches a
boolean literal. If it matches an integer literal, the name isn't set. To
account for this, the result struct's `bar` attribute is an option type:

```rust
...
if let Some(result) = my_pattern_alt(expr) {
    result.bar        // type: Option<&bool>
}
```

It's also possible to use a name in multiple alternation branches if they have
compatible types:

```rust
pattern!{
    // matches if expression is a boolean or integer literal
    my_pattern_mult: Expr =
        Lit(_#baz) | Array( Lit(_#baz) )
}
...
if let Some(result) = my_pattern_mult(expr) {
    result.baz        // type: &syntax::ast::Lit
}
```

Named submatches are a **flat** namespace and this is intended. In the example
above, two different sub-structures are assigned to a flat name. I expect that
for most lints, a flat namespace is sufficient and easier to work with than a
hierarchical one.

#### Two stages

Using named subpatterns, users can write lints in two stages. First, a coarse
selection of possible matches is produced by the pattern syntax. In the second
stage, the named subpattern references can be used to do additional tests like
asserting that a node hasn't been created as part of a macro expansion.

## Implementing clippy lints using patterns

As a "real-world" example, I re-implemented the `collapsible_if` lint using
patterns. The code can be found
[here](https://github.com/fkohlgrueber/rust-clippy-pattern/blob/039b07ecccaf96d6aa7504f5126720d2c9cceddd/clippy_lints/src/collapsible_if.rs#L88-L163).
The pattern-based version passes all test cases that were written for
`collapsible_if`.


# Reference-level explanation

## Overview

The following diagram shows the dependencies between the main parts of the
proposed solution:

```
                          Pattern syntax
                                |
                                |  parsing / lowering
                                v
                           PatternTree
                                ^
                                |
                                |
                          IsMatch trait
                                |
                                |
             +---------------+-----------+---------+
             |               |           |         |
             v               v           v         v
        syntax::ast     rustc::hir      syn       ...
```

The pattern syntax described in the previous section is parsed / lowered into
the so-called *PatternTree* data structure that represents a valid syntax tree
pattern. Matching a *PatternTree* against an actual syntax tree (e.g. rust ast /
hir or the syn ast, ...) is done using the *IsMatch* trait.

The *PatternTree* and the *IsMatch* trait are introduced in more detail in the
following sections.

## PatternTree

The core data structure of this RFC is the **PatternTree**.

It's a data structure similar to rust's AST / HIR, but with the following
differences:

- The PatternTree doesn't contain parsing information like `Span`s
- The PatternTree can represent alternatives, sequences and optionals

The code below shows a simplified version of the current PatternTree:

> Note: The current implementation can be found
> [here](https://github.com/fkohlgrueber/pattern-matching/blob/dfb3bc9fbab69cec7c91e72564a63ebaa2ede638/pattern-match/src/pattern_tree.rs#L50-L96).


```rust
pub enum Expr {
    Lit(Alt<Lit>),
    Array(Seq<Expr>),
    Block_(Alt<BlockType>),
    If(Alt<Expr>, Alt<BlockType>, Opt<Expr>),
    IfLet(
        Alt<BlockType>,
        Opt<Expr>,
    ),
}

pub enum Lit {
    Char(Alt<char>),
    Bool(Alt<bool>),
    Int(Alt<u128>),
}

pub enum Stmt {
    Expr(Alt<Expr>),
    Semi(Alt<Expr>),
}

pub enum BlockType {
    Block(Seq<Stmt>),
}
```

The `Alt`, `Seq` and `Opt` structs look like these:

> Note: The current implementation can be found
> [here](https://github.com/fkohlgrueber/pattern-matching/blob/dfb3bc9fbab69cec7c91e72564a63ebaa2ede638/pattern-match/src/matchers.rs#L35-L60).

```rust
pub enum Alt<T> {
    Any,
    Elmt(Box<T>),
    Alt(Box<Self>, Box<Self>),
    Named(Box<Self>, ...)
}

pub enum Opt<T> {
    Any,  // anything, but not None
    Elmt(Box<T>),
    None,
    Alt(Box<Self>, Box<Self>),
    Named(Box<Self>, ...)
}

pub enum Seq<T> {
    Any,
    Empty,
    Elmt(Box<T>),
    Repeat(Box<Self>, RepeatRange),
    Seq(Box<Self>, Box<Self>),
    Alt(Box<Self>, Box<Self>),
    Named(Box<Self>, ...)
}

pub struct RepeatRange {
    pub start: usize,
    pub end: Option<usize>  // exclusive
}
```

## Parsing / Lowering

The input of a `pattern!` macro call is parsed into a `ParseTree` first and then
lowered to a `PatternTree`.

Valid patterns depend on the *PatternTree* definitions. For example, the pattern
`Lit(Bool(_)*)` isn't valid because the parameter type of the `Lit` variant of
the `Expr` enum is `Any<Lit>` and therefore doesn't support repetition (`*`). As
another example, `Array( Lit(_)* )` is a valid pattern because the parameter of
`Array` is of type `Seq<Expr>` which allows sequences and repetitions.

> Note: names in the pattern syntax correspond to *PatternTree* enum
> **variants**. For example, the `Lit` in the pattern above refers to the `Lit`
> variant of the `Expr` enum (`Expr::Lit`), not the `Lit` enum.

## The IsMatch Trait

The pattern syntax and the *PatternTree* are independent of specific syntax tree
implementations (rust ast / hir, syn, ...). When looking at the different
pattern examples in the previous sections, it can be seen that the patterns
don't contain any information specific to a certain syntax tree implementation.
In contrast, clippy lints currently match against ast / hir syntax tree nodes
and therefore directly depend on their implementation.

The connection between the *PatternTree* and specific syntax tree
implementations is the `IsMatch` trait. It defines how to match *PatternTree*
nodes against specific syntax tree nodes. A simplified implementation of the
`IsMatch` trait is shown below:

```rust
pub trait IsMatch<O> {
    fn is_match(&self, other: &'o O) -> bool;
}
```

This trait needs to be implemented on each enum of the *PatternTree* (for the
corresponding syntax tree types). For example, the `IsMatch` implementation for
matching `ast::LitKind` against the *PatternTree's* `Lit` enum might look like
this:

```rust
impl IsMatch<ast::LitKind> for Lit {
    fn is_match(&self, other: &ast::LitKind) -> bool {
        match (self, other) {
            (Lit::Char(i), ast::LitKind::Char(j)) => i.is_match(j),
            (Lit::Bool(i), ast::LitKind::Bool(j)) => i.is_match(j),
            (Lit::Int(i), ast::LitKind::Int(j, _)) => i.is_match(j),
            _ => false,
        }
    }
}
```

All `IsMatch` implementations for matching the current *PatternTree* against
`syntax::ast` can be found
[here](https://github.com/fkohlgrueber/pattern-matching/blob/dfb3bc9fbab69cec7c91e72564a63ebaa2ede638/pattern-match/src/ast_match.rs).


# Drawbacks

#### Performance

The pattern matching code is currently not optimized for performance, so it
might be slower than hand-written matching code. Additionally, the two-stage
approach (matching against the coarse pattern first and checking for additional
properties later) might be slower than the current practice of checking for
structure and additional properties in one pass. For example, the following lint

```rust
pattern!{
    pat_if_without_else: Expr =
        If(
            _,
            Block(
                Expr( If(_, _, ())#inner )
                | Semi( If(_, _, ())#inner )
            )#then,
            ()
        )
}
...
fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
    if let Some(result) = pat_if_without_else(expr) {
        if !block_starts_with_comment(cx, result.then) {
            ...
        }
}
```

first matches against the pattern and then checks that the `then` block doesn't
start with a comment. Using clippy's current approach, it's possible to check
for these conditions earlier:

```rust
fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
    if_chain! {
        if let ast::ExprKind::If(ref check, ref then, None) = expr.node;
        if !block_starts_with_comment(cx, then);
        if let Some(inner) = expr_block(then);
        if let ast::ExprKind::If(ref check_inner, ref content, None) = inner.node;
        then {
            ...
        }
    }
}
```

Whether or not this causes performance regressions depends on actual patterns.
If it turns out to be a problem, the pattern matching algorithms could be
extended to allow "early filtering" (see the [Early Filtering](#early-filtering)
section in Future Possibilities).

That being said, I don't see any conceptual limitations regarding pattern
matching performance.

#### Applicability

Even though I'd expect that a lot of lints can be written using the proposed
pattern syntax, it's unlikely that all lints can be expressed using patterns. I
suspect that there will still be lints that need to be implemented by writing
custom pattern matching code. This would lead to mix within clippy's codebase
where some lints are implemented using patterns and others aren't. This
inconsistency might be considered a drawback.


# Rationale and alternatives

Specifying lints using syntax tree patterns has a couple of advantages compared
to the current approach of manually writing matching code. First, syntax tree
patterns allow users to describe patterns in a simple and expressive way. This
makes it easier to write new lints for both novices and experts and also makes
reading / modifying existing lints simpler.

Another advantage is that lints are independent of specific syntax tree
implementations (e.g. AST / HIR, ...). When these syntax tree implementations
change, only the `IsMatch` trait implementations need to be adapted and existing
lints can remain unchanged. This also means that if the `IsMatch` trait
implementations were integrated into the compiler, updating the `IsMatch`
implementations would be required for the compiler to compile successfully. This
could reduce the number of times clippy breaks because of changes in the
compiler. Another advantage of the pattern's independence is that converting an
`EarlyLintPass` lint into a `LatePassLint` wouldn't require rewriting the whole
pattern matching code. In fact, the pattern might work just fine without any
adaptions.


## Alternatives

### Rust-like pattern syntax

The proposed pattern syntax requires users to know the structure of the
`PatternTree` (which is very similar to the AST's / HIR's structure) and also
the pattern syntax. An alternative would be to introduce a pattern syntax that
is similar to actual Rust syntax (probably like the `quote!` macro). For
example, a pattern that matches `if` expressions that have `false` in their
condition could look like this:

```rust
if false {
    #[*]
}
```

#### Problems

Extending Rust syntax (which is quite complex by itself) with additional syntax
needed for specifying patterns (alternations, sequences, repetitions, named
submatches, ...) might become difficult to read and really hard to parse
properly.

For example, a pattern that matches a binary operation that has `0` on both
sides might look like this:

```
0 #[*:BinOpKind] 0
```

Now consider this slightly more complex example:

```
1 + 0 #[*:BinOpKind] 0
```

The parser would need to know the precedence of `#[*:BinOpKind]` because it
affects the structure of the resulting AST. `1 + 0 + 0` is parsed as `(1 + 0) +
0` while `1 + 0 * 0` is parsed as `1 + (0 * 0)`. Since the pattern could be any
`BinOpKind`, the precedence cannot be known in advance.

Another example of a problem would be named submatches. Take a look at this
pattern:

```rust
fn test() {
    1 #foo
}
```

Which node is `#foo` referring to? `int`, `ast::Lit`, `ast::Expr`, `ast::Stmt`?
Naming subpatterns in a rust-like syntax is difficult because a lot of AST nodes
don't have a syntactic element that can be used to put the name tag on. In these
situations, the only sensible option would be to assign the name tag to the
outermost node (`ast::Stmt` in the example above), because the information of
all child nodes can be retrieved through the outermost node. The problem with
this then would be that accessing inner nodes (like `ast::Lit`) would again
require manual pattern matching.

In general, Rust syntax contains a lot of code structure implicitly. This
structure is reconstructed during parsing (e.g. binary operations are
reconstructed using operator precedence and left-to-right) and is one of the
reasons why parsing is a complex task. The advantage of this approach is that
writing code is simpler for users.

When writing *syntax tree patterns*, each element of the hierarchy might have
alternatives, repetitions, etc.. Respecting that while still allowing
human-friendly syntax that contains structure implicitly seems to be really
complex, if not impossible.

Developing such a syntax would also require to maintain a custom parser that is
at least as complex as the Rust parser itself. Additionally, future changes in
the Rust syntax might be incompatible with such a syntax.

In summary, I think that developing such a syntax would introduce a lot of
complexity to solve a relatively minor problem.

The issue of users not knowing about the *PatternTree* structure could be solved
by a tool that, given a rust program, generates a pattern that matches only this
program (similar to the clippy author lint).

For some simple cases (like the first example above), it might be possible to
successfully mix Rust and pattern syntax. This space could be further explored
in a future extension.

# Prior art

The pattern syntax is heavily inspired by regular expressions (repetitions,
alternatives, sequences, ...).

From what I've seen until now, other linters also implement lints that directly
work on syntax tree data structures, just like clippy does currently. I would
therefore consider the pattern syntax to be *new*, but please correct me if I'm
wrong.

# Unresolved questions

#### How to handle multiple matches?

When matching a syntax tree node against a pattern, there are possibly multiple
ways in which the pattern can be matched. A simple example of this would be the
following pattern:

```rust
pattern!{
    my_pattern: Expr =
        Array( _* Lit(_)+#literals)
}
```

This pattern matches arrays that end with at least one literal. Now given the
array `[x, 1, 2]`, should `1` be matched as part of the `_*` or the `Lit(_)+`
part of the pattern? The difference is important because the named submatch
`#literals` would contain 1 or 2 elements depending how the pattern is matched.
In regular expressions, this problem is solved by matching "greedy" by default
and "non-greedy" optionally.

I haven't looked much into this yet because I don't know how relevant it is for
most lints. The current implementation simply returns the first match it finds.

# Future possibilities

#### Implement rest of Rust Syntax

The current project only implements a small part of the Rust syntax. In the
future, this should incrementally be extended to more syntax to allow
implementing more lints. Implementing more of the Rust syntax requires extending
the `PatternTree` and `IsMatch` implementations, but should be relatively
straight-forward.

#### Early filtering

As described in the *Drawbacks/Performance* section, allowing additional checks
during the pattern matching might be beneficial.

The pattern below shows how this could look like:

```rust
pattern!{
    pat_if_without_else: Expr =
        If(
            _,
            Block(
                Expr( If(_, _, ())#inner )
                | Semi( If(_, _, ())#inner )
            )#then,
            ()
        )
    where
        !in_macro(#then.span);
}
```

The difference compared to the currently proposed two-stage filtering is that
using early filtering, the condition (`!in_macro(#then.span)` in this case)
would be evaluated as soon as the `Block(_)#then` was matched.

Another idea in this area would be to introduce a syntax for backreferences.
They could be used to require that multiple parts of a pattern should match the
same value. For example, the `assign_op_pattern` lint that searches for `a = a
op b` and recommends changing it to `a op= b` requires that both occurrences of
`a` are the same. Using `=#...` as syntax for backreferences, the lint could be
implemented like this:

```rust
pattern!{
    assign_op_pattern: Expr =
        Assign(_#target, Binary(_, =#target, _)
}
```

#### Match descendant

A lot of lints currently implement custom visitors that check whether any
subtree (which might not be a direct descendant) of the current node matches
some properties. This cannot be expressed with the proposed pattern syntax.
Extending the pattern syntax to allow patterns like "a function that contains at
least two return statements" could be a practical addition.

#### Negation operator for alternatives

For patterns like "a literal that is not a boolean literal" one currently needs
to list all alternatives except the boolean case. Introducing a negation
operator that allows to write `Lit(!Bool(_))` might be a good idea. This pattern
would be equivalent to `Lit( Char(_) | Int(_) )` (given that currently only three
literal types are implemented).

#### Functional composition

Patterns currently don't have any concept of composition. This leads to
repetitions within patterns. For example, one of the collapsible-if patterns
currently has to be written like this:

```rust
pattern!{
    pat_if_else: Expr =
        If(
            _,
            _,
            Block_(
                Block(
                    Expr((If(_, _, _?) | IfLet(_, _?))#else_) |
                    Semi((If(_, _, _?) | IfLet(_, _?))#else_)
                )#block_inner
            )#block
        ) |
        IfLet(
            _,
            Block_(
                Block(
                    Expr((If(_, _, _?) | IfLet(_, _?))#else_) |
                    Semi((If(_, _, _?) | IfLet(_, _?))#else_)
                )#block_inner
            )#block
        )
}
```

If patterns supported defining functions of subpatterns, the code could be
simplified as follows:

```rust
pattern!{
    fn expr_or_semi(expr: Expr) -> Stmt {
        Expr(expr) | Semi(expr)
    }
    fn if_or_if_let(then: Block, else: Opt<Expr>) -> Expr {
        If(_, then, else) | IfLet(then, else)
    }
    pat_if_else: Expr =
        if_or_if_let(
            _,
            Block_(
                Block(
                    expr_or_semi( if_or_if_let(_, _?)#else_ )
                )#block_inner
            )#block
        )
}
```

Additionally, common patterns like `expr_or_semi` could be shared between
different lints.

#### Clippy Pattern Author

Another improvement could be to create a tool that, given some valid Rust
syntax, generates a pattern that matches this syntax exactly. This would make
starting to write a pattern easier. A user could take a look at the patterns
generated for a couple of Rust code examples and use that information to write a
pattern that matches all of them.

This is similar to clippy's author lint.

#### Supporting other syntaxes

Most of the proposed system is language-agnostic. For example, the pattern
syntax could also be used to describe patterns for other programming languages.

In order to support other languages' syntaxes, one would need to implement
another `PatternTree` that sufficiently describes the languages' AST and
implement `IsMatch` for this `PatternTree` and the languages' AST.

One aspect of this is that it would even be possible to write lints that work on
the pattern syntax itself. For example, when writing the following pattern


```rust
pattern!{
    my_pattern: Expr =
        Array( Lit(Bool(false)) Lit(Bool(false)) )
}
```

a lint that works on the pattern syntax's AST could suggest using this pattern
instead:

```rust
pattern!{
    my_pattern: Expr =
        Array( Lit(Bool(false)){2} )
}
```

In the future, clippy could use this system to also provide lints for custom
syntaxes like those found in macros.
