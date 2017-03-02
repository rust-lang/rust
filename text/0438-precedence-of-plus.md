- Start Date: 2014-11-18
- RFC PR: [rust-lang/rfcs#438](https://github.com/rust-lang/rfcs/pull/438)
- Rust Issue: [rust-lang/rust#19092](https://github.com/rust-lang/rust/issues/19092)

# Summary

Change the precedence of `+` (object bounds) in type grammar so that
it is similar to the precedence in the expression grammars.

# Motivation

Currently `+` in types has a much higher precedence than it does in expressions.
This means that for example one can write a type like the following:

```
&Object+Send
```
    
Whereas if that were an expression, parentheses would be required:

```rust
&(Object+Send)
````
    
Besides being confusing in its own right, this loose approach with
regard to precedence yields ambiguities with unboxed closure bounds:

```rust
fn foo<F>(f: F)
    where F: FnOnce(&int) -> &Object + Send
{ }
```

In this example, it is unclear whether `F` returns an object which is
`Send`, or whether `F` itself is `Send`.

# Detailed design

This RFC proposes that the precedence of `+` be made lower than unary
type operators. In addition, the grammar is segregated such that in
"open-ended" contexts (e.g., after `->`), parentheses are required to
use a `+`, whereas in others (e.g., inside `<>`), parentheses are not.
Here are some examples:

```rust
// Before                             After                         Note
// ~~~~~~                             ~~~~~                         ~~~~
   &Object+Send                       &(Object+Send)
   &'a Object+'a                      &'a (Object+'a)
   Box<Object+Send>                   Box<Object+Send>
   foo::<Object+Send,int>(...)        foo::<Object+Send,int>(...)
   Fn() -> Object+Send                Fn() -> (Object+Send)         // (*)
   Fn() -> &Object+Send               Fn() -> &(Object+Send)
   
// (*) Must yield a type error, as return type must be `Sized`.
```

More fully, the type grammar is as follows (EBNF notation):

    TYPE = PATH
         | '&' [LIFETIME] TYPE
         | '&' [LIFETIME] 'mut' TYPE
         | '*' 'const' TYPE
         | '*' 'mut' TYPE
         | ...
         | '(' SUM ')'
    SUM  = TYPE { '+' TYPE }
    PATH = IDS '<' SUM { ',' SUM } '>'
         | IDS '(' SUM { ',' SUM } ')' '->' TYPE
    IDS  = ['::'] ID { '::' ID }

Where clauses would use the following grammar:

    WHERE_CLAUSE = PATH { '+' PATH }
    
One property of this grammar is that the `TYPE` nonterminal does not
require a terminator as it has no "open-ended" expansions. `SUM`, in
contrast, can be extended any number of times via the `+` token. Hence
is why `SUM` must be enclosed in parens to make it into a `TYPE`.
    
# Drawbacks

Common types like `&'a Foo+'a` become slightly longer (`&'a (Foo+'a)`).

# Alternatives

We could live with the inconsistency between the type/expression
grammars and disambiguate where clauses in an ad-hoc way.

# Unresolved questions

None.
