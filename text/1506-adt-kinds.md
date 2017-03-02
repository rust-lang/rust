- Feature Name: clarified_adt_kinds
- Start Date: 2016-02-07
- RFC PR: https://github.com/rust-lang/rfcs/pull/1506
- Rust Issue: https://github.com/rust-lang/rust/issues/35626

# Summary
[summary]: #summary

Provide a simple model describing three kinds of structs and variants and their relationships.  
Provide a way to match on structs/variants in patterns regardless of their kind (`S{..}`).  
Permit tuple structs and tuple variants with zero fields (`TS()`).

# Motivation
[motivation]: #motivation

There's some mental model lying under the current implementation of ADTs, but it is not written
out explicitly and not implemented completely consistently.
Writing this model out helps to identify its missing parts.
Some of this missing parts turn out to be practically useful.
This RFC can also serve as a piece of documentation.

# Detailed design
[design]: #detailed-design

The text below mostly talks about structures, but almost everything is equally applicable to
variants.

## Braced structs

Braced structs are declared with braces (unsurprisingly).

```
struct S {
    field1: Type1,
    field2: Type2,
    field3: Type3,
}
```

Braced structs are the basic struct kind, other kinds are built on top of them.
Braced structs have 0 or more user-named fields and are defined only in type namespace.

Braced structs can be used in struct expressions `S{field1: expr, field2: expr}`, including
functional record update (FRU) `S{field1: expr, ..s}`/`S{..s}` and with struct patterns
`S{field1: pat, field2: pat}`/`S{field1: pat, ..}`/`S{..}`.
In all cases the path `S` of the expression or pattern is looked up in the type namespace (so these
expressions/patterns can be used with type aliases).
Fields of a braced struct can be accessed with dot syntax `s.field1`.

Note: struct *variants* are currently defined in the value namespace in addition to type namespace,
 there are no particular reasons for this and this is probably temporary.

## Unit structs

Unit structs are defined without any fields or brackets.

```
struct US;
```

Unit structs can be thought of as a single declaration for two things: a basic struct

```
struct US {}
```

and a constant with the same name<sup>Note 1</sup>

```
const US: US = US{};
```

Unit structs have 0 fields and are defined in both type (the type `US`) and value (the
constant `US`) namespaces.

As a basic struct, a unit struct can participate in struct expressions `US{}`, including FRU
`US{..s}` and in struct patterns `US{}`/`US{..}`. In both cases the path `US` of the expression
or pattern is looked up in the type namespace (so these expressions/patterns can be used with type
aliases).
Fields of a unit struct could also be accessed with dot syntax, but it doesn't have any fields.

As a constant, a unit struct can participate in unit struct expressions `US` and unit struct
patterns `US`, both of these are looked up in the value namespace in which the constant `US` is
defined (so these expressions/patterns cannot be used with type aliases).

Note 1: the constant is not exactly a `const` item, there are subtle differences (e.g. with regards
to `match` exhaustiveness), but it's a close approximation.  
Note 2: the constant is pretty weirdly namespaced in case of unit *variants*, constants can't be
defined in "enum modules" manually.

## Tuple structs

Tuple structs are declared with parentheses.
```
struct TS(Type0, Type1, Type2);
```

Tuple structs can be thought of as a single declaration for two things: a basic struct

```
struct TS {
    0: Type0,
    1: Type1,
    2: Type2,
}
```

and a constructor function with the same name<sup>Note 2</sup>

```
fn TS(arg0: Type0, arg1: Type1, arg2: Type2) -> TS {
    TS{0: arg0, 1: arg1, 2: arg2}
}
```

Tuple structs have 0 or more automatically-named fields and are defined in both type (the type `TS`)
and the value (the constructor function `TS`) namespaces.

As a basic struct, a tuple struct can participate in struct expressions `TS{0: expr, 1: expr}`,
including FRU `TS{0: expr, ..ts}`/`TS{..ts}` and in struct patterns
`TS{0: pat, 1: pat}`/`TS{0: pat, ..}`/`TS{..}`.
In both cases the path `TS` of the expression or pattern is looked up in the type namespace (so
these expressions/patterns can be used with type aliases).
Fields of a tuple struct can be accessed with dot syntax `ts.0`.

As a constructor, a tuple struct can participate in tuple struct expressions `TS(expr, expr)` and
tuple struct patterns `TS(pat, pat)`/`TS(..)`, both of these are looked up in the value namespace
in which the constructor `TS` is defined (so these expressions/patterns cannot be used with type
aliases). Tuple struct expressions `TS(expr, expr)` are usual
function calls, but the compiler reserves the right to make observable improvements to them based
on the additional knowledge, that `TS` is a constructor.

Note 1: the automatically assigned field names are quite interesting, they are not identifiers
lexically (they are integer literals), so such fields can't be defined manually.  
Note 2: the constructor function is not exactly a `fn` item, there are subtle differences (e.g. with
regards to privacy checks), but it's a close approximation.

## Summary of the changes.

Everything related to braced structs and unit structs is already implemented.

New: Permit tuple structs and tuple variants with 0 fields. This restriction is artificial and can
be lifted trivially. Macro writers dealing with tuple structs/variants will be happy to get rid of
this one special case.

New: Permit using tuple structs and tuple variants in braced struct patterns and expressions not
requiring naming their fields - `TS{..ts}`/`TS{}`/`TS{..}`. This doesn't require much effort to
implement as well.  
This also means that `S{..}` patterns can be used to match structures and variants of any kind.
The desire to have such "match everything" patterns is sometimes expressed given
that number of fields in structures and variants can change from zero to non-zero and back during
development.  
An extra benefit is ability to match/construct tuple structs using their type aliases.

New: Permit using tuple structs and tuple variants in braced struct patterns and expressions
requiring naming their fields - `TS{0: expr}`/`TS{0: pat}`/etc.
While this change is important for consistency, there's not much motivation for it in hand-written
code besides shortening patterns like `ItemFn(_, _, unsafety, _, _, _)` into something like
`ItemFn{2: unsafety, ..}` and ability to match/construct tuple structs using their type aliases.  
However, automatic code generators (e.g. syntax extensions) can get more benefits from the
ability to generate uniform code for all structure kinds.  
`#[derive]` for example, currently has separate code paths for generating expressions and patterns
for braces structs (`ExprStruct`/`PatKind::Struct`), tuple structs
(`ExprCall`/`PatKind::TupleStruct`) and unit structs (`ExprPath`/`PatKind::Path`). With proposed
changes `#[derive]` could simplify its logic and always generate braced forms for expressions and
patterns.

# Drawbacks
[drawbacks]: #drawbacks

None.

# Alternatives
[alternatives]: #alternatives

None.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
