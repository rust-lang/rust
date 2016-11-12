- Feature Name: field-init-shorthand
- Start Date: 2016-07-18
- RFC PR: https://github.com/rust-lang/rfcs/pull/1682
- Rust Issue: https://github.com/rust-lang/rust/issues/37340

# Summary
[summary]: #summary

When initializing a data structure (struct, enum, union) with named fields,
allow writing `fieldname` as a shorthand for `fieldname: fieldname`. This
allows a compact syntax for initialization, with less duplication.

Example usage:

    struct SomeStruct { field1: ComplexType, field2: AnotherType }

    impl SomeStruct {
        fn new() -> Self {
            let field1 = {
                // Various initialization code
            };
            let field2 = {
                // More initialization code
            };
            SomeStruct { field1, field2 }
        }
    }

# Motivation
[motivation]: #motivation

When writing initialization code for a data structure, the names of the
structure fields often become the most straightforward names to use for their
initial values as well. At the end of such an initialization function, then,
the initializer will contain many patterns of repeated field names as field
values: `field: field, field2: field2, field3: field3`.

Such repetition of the field names makes it less ergonomic to separately
declare and initialize individual fields, and makes it tempting to instead
embed complex code directly in the initializer to avoid repetition.

Rust already allows
[similar syntax for destructuring in pattern matches](https://doc.rust-lang.org/book/patterns.html#destructuring):
a pattern match can use `SomeStruct { field1, field2 } => ...` to match
`field1` and `field2` into values with the same names. This RFC introduces
symmetrical syntax for initializers.

A family of related structures will often use the same field name for a
semantically-similar value. Combining this new syntax with the existing
pattern-matching syntax allows simple movement of data between fields with a
pattern match: `Struct1 { field1, .. } => Struct2 { field1 }`.

The proposed syntax also improves structure initializers in closures, such as
might appear in a chain of iterator adapters: `|field1, field2| SomeStruct {
field1, field2 }`.

This RFC takes inspiration from the Haskell
[NamedFieldPuns extension](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/glasgow_exts.html#record-puns),
and from ES6
[shorthand property names](http://www.ecma-international.org/ecma-262/6.0/#sec-object-initializer).

# Detailed design
[design]: #detailed-design

## Grammar

In the initializer for a `struct` with named fields, a `union` with named
fields, or an enum variant with named fields, accept an identifier `field` as a
shorthand for `field: field`.

With reference to the grammar in `parser-lalr.y`, this proposal would
expand the `field_init`
[rule](https://github.com/rust-lang/rust/blob/master/src/grammar/parser-lalr.y#L1663-L1665)
to the following:

    field_init
    : ident
    | ident ':' expr
    ;

## Interpretation

The shorthand initializer `field` always behaves in every possible way like the
longhand initializer `field: field`. This RFC introduces no new behavior or
semantics, only a purely syntactic shorthand. The rest of this section only
provides further examples to explicitly clarify that this new syntax remains
entirely orthogonal to other initializer behavior and semantics.

## Examples

If the struct `SomeStruct` has fields `field1` and `field2`, the initializer
`SomeStruct { field1, field2 }` behaves in every way like the initializer
`SomeStruct { field1: field1, field2: field2 }`.

An initializer may contain any combination of shorthand and full field
initializers:

    let a = SomeStruct { field1, field2: expression, field3 };
    let b = SomeStruct { field1: field1, field2: expression, field3: field3 };
    assert_eq!(a, b);

An initializer may use shorthand field initializers together with
[update syntax](https://doc.rust-lang.org/book/structs.html#update-syntax):

    let a = SomeStruct { field1, .. someStructInstance };
    let b = SomeStruct { field1: field1, .. someStructInstance };
    assert_eq!(a, b);

## Compilation errors

This shorthand initializer syntax does not introduce any new compiler errors
that cannot also occur with the longhand initializer syntax `field: field`.
Existing compiler errors that can occur with the longhand initializer syntax
`field: field` also apply to the shorthand initializer syntax `field`:

- As with the longhand initializer `field: field`, if the structure has no
  field with the specified name `field`, the shorthand initializer `field`
  results in a compiler error for attempting to initialize a non-existent
  field.

- As with the longhand initializer `field: field`, repeating a field name
  within the same initializer results in a compiler error
  ([E0062](https://doc.rust-lang.org/error-index.html#E0062)); this occurs with
  any combination of shorthand initializers or full `field: expression`
  initializers.

- As with the longhand initializer `field: field`, if the name `field` does not
  resolve, the shorthand initializer `field` results in a compiler error for an
  unresolved name ([E0425](https://doc.rust-lang.org/error-index.html#E0425)).

- As with the longhand initializer `field: field`, if the name `field` resolves
  to a value with type incompatible with the field `field` in the structure,
  the shorthand initializer `field` results in a compiler error for mismatched
  types ([E0308](https://doc.rust-lang.org/error-index.html#E0308)).

# Drawbacks
[drawbacks]: #drawbacks

This new syntax could significantly improve readability given clear and local
field-punning variables, but could also be abused to decrease readability if
used with more distant variables.

As with many syntactic changes, a macro could implement this instead. See the
Alternatives section for discussion of this.

The shorthand initializer syntax looks similar to positional initialization of
a structure without field names; reinforcing this, the initializer will
commonly list the fields in the same order that the struct declares them.
However, the shorthand initializer syntax differs from the positional
initializer syntax (such as for a tuple struct) in that the positional syntax
uses parentheses instead of braces: `SomeStruct(x, y)` is unambiguously a
positional initializer, while `SomeStruct { x, y }` is unambiguously a
shorthand initializer for the named fields `x` and `y`.

# Alternatives
[alternatives]: #alternatives

## Wildcards

In addition to this syntax, initializers could support omitting the field names
entirely, with syntax like `SomeStruct { .. }`, which would implicitly
initialize omitted fields from identically named variables. However, that would
introduce far too much magic into initializers, and the context-dependence
seems likely to result in less readable, less obvious code.

## Macros

A macro wrapped around the initializer could implement this syntax, without
changing the language; for instance, `pun! { SomeStruct { field1, field2 } }`
could expand to `SomeStruct { field1: field1, field2: field2 }`. However, this
change exists to make structure construction shorter and more expressive;
having to use a macro would negate some of the benefit of doing so,
particularly in places where brevity improves readability, such as in a closure
in the middle of a larger expression. There is also precedent for
language-level support. Pattern matching already allows using field names as
the _destination_ for the field values via destructuring. This change adds a
symmetrical mechanism for construction which uses existing names as _sources_.

## Sigils

To minimize confusing shorthand expressions with the construction of
tuple-like structs, we might elect to prefix expanded field names with
sigils.

For example, if the sigil were `:`, the existing syntax `S { x: x }`
would be expressed as `S { :x }`. This is used in
[MoonScript](http://moonscript.org/reference/#the-language/table-literals).

This particular choice of sigil may be confusing, due to the
already-overloaded use of `:` for fields and type ascription. Additionally,
in languages such as Ruby and Elixir, `:x` denotes a symbol or atom, which
may be confusing for newcomers.

Other sigils could be used instead, but even then we are then increasing
the amount of new syntax being introduced. This both increases language
complexity and reduces the gained compactness, worsening the
cost/benefit ratio of adding a shorthand. Any use of a sigil also breaks
the symmetry between binding pattern matching and the proposed
shorthand.

## Keyword-prefixed

Similarly to sigils, we could use a keyword like Nix uses
[inherit](http://nixos.org/nix/manual/#idm46912467627696). Some forms we could
decide upon (using `use` as the keyword of choice here, but it could be
something else), it could look like the following.

* `S { use x, y, z: 10}`
* `S { use (x, y), z: 10 }`
* `S { use {x, y}, z: 10 }`
* `S { use x, use y, z: 10}`

This has the same drawbacks as sigils except that it won't be confused for
symbols in other languages or adding more sigils. It also has the benefit
of being something that can be searched for in documentation.
