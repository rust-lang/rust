- Feature Name: attributes_with_literals
- Start Date: 2016-03-28
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

This RFC proposes accepting literals in attributes by defining the grammar of attributes as:

```ebnf
attr : '#' '[' meta_item ']' ;

meta_item : IDENT ( '=' LIT | '(' meta_item_inner? ')' )? ;

meta_item_inner : (meta_item | LIT) (',' meta_item_inner)? ;
```

Note that `LIT` is a valid Rust literal and `IDENT` is a valid Rust identifier. The following
attributes, among others, would be accepted by this grammar:

```rust
#[attr]
#[attr()]
#[attr(ident)]
#[attr(ident, ident = 100, ident = "hello", ident(100))]
#[attr(100)]
#[attr("hello")]
#[repr(C, align = 4)]
#[repr(C, align(4))]
```

# Motivation
[motivation]: #motivation

At present, literals are only accepted as the value of a key-value pair in attributes. What's more,
only _string_ literals are accepted. This means that literals can only appear in forms of
`#[attr(name = "value")]` or `#[attr = "value"]`.

This forces non-string literal values to be awkwardly stringified. For example, while it is clear
that something like alignment should be an integer value, the following are disallowed:
`#[align(4)]`, `#[align = 4]`. Instead, we must use something akin to `#[align = "4"]`. Even
`#[align("4")]` and `#[name("name")]` are disallowed, forcing identifiers or key-values to be used
instead: `#[align(size = "4")]` or `#[name(name)]`.

In short, the current design forces users to use values of the wrong type in attributes.

### Cleaner Attributes

Implementation of this RFC can clean up the following attributes in the standard library:

* `#![recursion_limit = "64"]` **=>** `#![recursion_limit = 64]` or `#![recursion_limit(64)]`
* `#[cfg(all(unix, target_pointer_width = "32"))]` **=>** `#[cfg(all(unix, target_pointer_width = 32))]`

If `align` were to be added as an attribute, the following are now valid options for its syntax:

* `#[repr(align(4))]`
* `#[repr(align = 4)]`
* `#[align = 4]`
* `#[align(4)]`

### Syntax Extensions

As syntax extensions mature and become more widely used, being able to use literals in a variety of
positions becomes more important.

# Detailed design
[design]: #detailed-design

1.  The `MetaItemKind` structure would need to allow literals as top-level entities:

     ```rust
     pub enum MetaItemKind {
         Word(InternedString),
         List(InternedString, Vec<P<MetaItem>>),
         NameValue(InternedString, Lit),
         Lit,
     }
     ```

2.  `libsyntax` (`libsyntax/parse/attr.rs`) would need to be modified to allow literals as values in
    k/v pairs and as top-level entities of a list.

3.  Crate metadata encoding/decoding would need to encode and decode literals in attributes.

# Drawbacks
[drawbacks]: #drawbacks

This RFC requires a change to the AST and is likely to break syntax extensions using attributes in
the wild.

# Alternatives
[alternatives]: #alternatives

### Token Trees

An alternative is to allow any tokens inside of an attribute. That is, the grammar could be:

```ebnf
attr : '#' '[' TOKEN+ ']' ;
```

where `TOKEN` is any valid Rust token. The drawback to this approach is that attributes lose any
sense of structure. This results in more difficult and verbose attribute parsing, although this
could be ameliorated through libraries. Further, this would require almost all of the existing
attribute parsing code to change.

The advantage, of course, is that it allows any syntax and is rather future proof. It is also more
inline with `macro!`s.

### Only Allow Literals as Values in K/V Pairs

Instead of allowing literals in top-level positions, i.e. `#[attr(4)]`, only allow them as values in
key value pairs: `#[attr = 4]` or `#[attr(ident = 4)]`. This has the nice advantage that it was the
initial idea for attributes, and so the AST types already reflect this. As such, no changes would
have to be made to existing code. The drawback, of course, is the lack of flexibility. `#[repr(C,
align(4))]` would no longer be valid.

### Do Nothing

Of course, the current design could be kept. Although it seems that the initial intention was for a
form of literals to be allowed. Unfortunately, this idea was [scrapped due to release pressure] and
never revisited. Even the manual alludes to allowing all literals.

  [scrapped due to release pressure]: https://github.com/rust-lang/rust/issues/623

# Unresolved questions
[unresolved]: #unresolved-questions

None that I can think of.
