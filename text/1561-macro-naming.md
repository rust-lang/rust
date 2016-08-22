- Feature Name: N/A (part of other unstable features)
- Start Date: 2016-02-11
- RFC PR: https://github.com/rust-lang/rfcs/pull/1561
- Rust Issue: https://github.com/rust-lang/rust/issues/35896

# Summary
[summary]: #summary

Naming and modularisation for macros.

This RFC proposes making macros a first-class citizen in the Rust module system.
Both macros by example (`macro_rules` macros) and procedural macros (aka syntax
extensions) would use the same naming and modularisation scheme as other items
in Rust.

For procedural macros, this RFC could be implemented immediately or as part of a
larger effort to reform procedural macros. For macros by example, this would be
part of a macros 2.0 feature, the rest of which will be described in a separate
RFC. This RFC depends on the changes to name resolution described in
[RFC 1560](https://github.com/rust-lang/rfcs/pull/1560).

# Motivation
[motivation]: #motivation

Currently, procedural macros are not modularised at all (beyond the crate
level). Macros by example have a [custom modularisation
scheme](https://github.com/rust-lang/rfcs/blob/master/text/0453-macro-reform.md)
which involves modules to some extent, but relies on source ordering and
attributes which are not used for other items. Macros cannot be imported or
named using the usual syntax. It is confusing that macros use their own system
for modularisation. It would be far nicer if they were a more regular feature of
Rust in this respect.


# Detailed design
[design]: #detailed-design

## Defining macros

This RFC does not propose changes to macro definitions. It is envisaged that
definitions of procedural macros will change, see [this blog post](http://ncameron.org/blog/macro-plans-syntax/)
for some rough ideas. I'm assuming that procedural macros will be defined in
some function-like way and that these functions will be defined in modules in
their own crate (to start with).

Ordering of macro definitions in the source text will no longer be significant.
A macro may be used before it is defined, as long as it can be named. That is,
macros follow the same rules regarding ordering as other items. E.g., this will
work:

```
foo!();

macro! foo { ... }
```

(Note, I'm using a hypothetical `macro!` defintion which I will define in a future
RFC. The reader can assume it works much like `macro_rules!`, but with the new
naming scheme).

Macro expansion order is also not defined by source order. E.g., in `foo!(); bar!();`,
`bar` may be expanded before `foo`. Ordering is only guaranteed as far as it is
necessary. E.g., if `bar` is only defined by expanding `foo`, then `foo` must be
expanded before `bar`.

## Function-like macro uses

A function-like macro use (c.f., attribute-like macro use) is a macro use which
uses `foo!(...)` or `foo! ident (...)` syntax (where `()` may also be `[]` or `{}`).

Macros may be named by using a `::`-separated path. Naming follows the same
rules as other items in Rust.

If a macro `baz` (by example or procedural) is defined in a module `bar` which
is nested in `foo`, then it may be used anywhere in the crate using an
absolute path: `::foo::bar::baz!(...)`. It can be used via relative paths in the
usual way, e.g., inside `foo` as `bar::baz!()`.

Macros declared inside a function body can only be used inside that function
body.

For procedural macros, the path must point to the function defining the macro.

The grammar for macros is changed, anywhere we currently parser `name "!"`, we
now parse `path "!"`. I don't think this introduces any issues.

Name lookup follows the same name resolution rules as other items. See [RFC
1560](https://github.com/rust-lang/rfcs/pull/1560) for details on how name
resolution could be adapted to support this.

## Attribute-like macro uses

Attribute macros may also be named using a `::`-separated path. Other than
appearing in an attribute, these also follow the usual Rust naming rules.

E.g., `#[::foo::bar::baz(...)]` and `#[bar::baz(...)]` are uses of absolute and
relative paths, respectively.


## Importing macros

Importing macros is done using `use` in the same way as other items. An `!` is
not necessary in an import item. Macros are imported into their own namespace
and do not shadow or overlap items with the same name in the type or value
namespaces.

E.g., `use foo::bar::baz;` imports the macro `baz` from the module `::foo::bar`.
Macro imports may be used in import lists (with other macro imports and with
non-macro imports).

Where a glob import (`use ...::*;`) imports names from a module including macro
definitions, the names of those macros are also imported. E.g., `use
foo::bar::*;` would import `baz` along with any other items in `foo::bar`.

Where macros are defined in a separate crate, these are imported in the same way
as other items by an `extern crate` item.

No `#[macro_use]` or `#[macro_export]` annotations are required.


## Shadowing

Macro names follow the same shadowing rules as other names. For example, an
explicitly declared macro would shadow a glob-imported macro with the same name.
Note that since macros are in a different namespace from types and values, a
macro cannot shadow a type or value or vice versa.


# Drawbacks
[drawbacks]: #drawbacks

If the new macro system is not well adopted by users, we could be left with two
very different schemes for naming macros depending on whether a macro is defined
by example or procedurally. That would be inconsistent and annoying. However, I
hope we can make the new macro system appealing enough and close enough to the
existing system that migration is both desirable and easy.


# Alternatives
[alternatives]: #alternatives

We could adopt the proposed scheme for procedural macros only and keep the
existing scheme for macros by example.

We could adapt the current macros by example scheme to procedural macros.

We could require the `!` in macro imports to distinguish them from other names.
I don't think this is necessary or helpful.

We could continue to require `macro_export` annotations on top of this scheme.
However, I prefer moving to a scheme using the same privacy system as the rest
of Rust, see below.


# Unresolved questions
[unresolved]: #unresolved-questions

## Privacy for macros

I would like that macros follow the same rules for privacy as other Rust items,
i.e., they are private by default and may be marked as `pub` to make them
public. This is not as straightforward as it sounds as it requires parsing `pub
macro! foo` as a macro definition, etc. I leave this for a separate RFC.

## Scoped attributes

It would be nice for tools to use scoped attributes as well as procedural
macros, e.g., `#[rustfmt::skip]` or `#[rust::new_attribute]`. I believe this
should be straightforward syntactically, but there are open questions around
when attributes are ignored or seen by tools and the compiler. Again, I leave it
for a future RFC.

## Inline procedural macros

Some day, I hope that procedural macros may be defined in the same crate in
which they are used. I leave the details of this for later, however, I don't
think this affects the design of naming - it should all Just Work.

## Applying to existing macros

This RFC is framed in terms of a new macro system. There are various ways that
some parts of it could be applied to existing macros (`macro_rules!`) to
backwards compatibly make existing macros usable under the new naming system.

I want to leave this question unanswered for now. Until we get some experience
implementing this feature it is unclear how much this is possible. Once we know
that we can try to decide how much of that is also desirable.
