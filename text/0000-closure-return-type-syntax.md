- Feature Name: N/A
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Restrict closure return type syntax for future compatibility.

# Motivation

Today's closure return type syntax juxtaposes a type and an
expression. This is dangerous: if we choose to extend the type grammar
to be more acceptable, we can easily break existing code.

# Detailed design

The current closure syntax for annotating the return type is `|Args|
-> Type Expr`, where `Type` is the return type and `Expr` is the body
of the closure. This syntax is future hostile and relies on being able
to determine the end point of a type. If we extend the syntax for
types, we could cause parse errors in existing code.

An example from history is that we extended the type grammar to
include things like `Fn(..)`. This would have caused the following,
previous, legal -- closure not to parse: `|| -> Foo (Foo)`. As a
simple fix, this RFC proposes that if a return type annotation is
supplied, the body must be enclosed in braces: `|| -> Foo { (Foo) }`.
Types are already juxtaposed with open braces in `fn` items, so this
should not be an additional danger for future evolution.

# Drawbacks

This design is minimally invasive but perhaps unfortunate in that it's
not obvious that braces would be required. But then, return type
annotations are very rarely used.

# Alternatives

I am not aware of any alternate designs. One possibility would be to
remove return type anotations altogether, perhaps relying on type
ascription or other annotations to force the inferencer to figure
things out, but they are useful in rare scenarios. In particular type
ascription would not be able to handle a higher-ranked signature like
`for<'a> &'a X -> &'a Y` without improving the type checker
implementation in other ways (in particular, we don't infer
generalization over lifetimes at present, unless we can figure it out
from the expected type or explicit annotations).

# Unresolved questions

None.
