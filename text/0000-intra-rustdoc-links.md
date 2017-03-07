- Feature Name: `intra_rustdoc_links`
- Start Date: 2017-03-06
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add a notation how to create relative links in documentation comments
(based on Rust item paths)
and extend Rustdoc to automatically turn this into working links.


# Motivation
[motivation]: #motivation

It is good practice in the Rust community to
add documentation to all public items of a crate,
as the API documentation as rendered by Rustdoc is the main documentation of most libaries.
Documentation comments at the module (or crate) level are used to
give an overview of the module (or crate)
and describe how the items of a crate can be used together.
To make navigating the documentation easy,
crate authors make these items link to their individual entries
in the API docs.

Currently, these links are plain Markdown links,
and the URLs are the (relative) paths of the items' pages
in the rendered Rustdoc output.
This is sadly very fragile in several ways:

1. As the same doc comment can be rendered on several Rustdoc pages
  and thus on separate directory levels
  (e.g., the summary page of a module, and a struct's own page),
  it is not possible to confidently use relative paths.
  For example,
  adding a link to `../foo/struct.Bar.html`
  to the first paragraph of the doc comment of the module `lorem`
  will work on the rendered `/lorem/index.html` page,
  but not on the crate's summary page `/index.html`.
2. Using absolute paths in links
  (like `/crate-name/foo/struct.Bar.html`)
  to circumvent the previous issue
  might work for the author's own hosted version,
  but will break when
  looking at the documentation using `cargo doc --open`
  (which uses `file:///` URLs)
  or when using docs.rs.
3. Should Rustdoc's file name scheme ever change
  (it has change before^(citation needed)),
  all manually created links need to be updated.

To solve this dilemma,
we propose extending Rustdoc
to be able to generate relative links that work in all contexts.


# Detailed design
[design]: #detailed-design

Markdown allows writing links in several forms:

1. `[Human readable name](URL)`
2. `[Human readable name][label]`,
  and somewhere else in the document: `[label]: URL`
3. `<URL>` which will be turned into the equivalent of `[URL](URL)`
  (some renderer will automatically prepend `http://` as a schema if necessary)

We prospose that
in each occurance of `URL` in the above list,
it should also be possible to write a Rust path
(as defined [in the reference][ref-paths]).

[ref-paths]: https://github.com/rust-lang-nursery/reference/blob/2d23ea601f017c106a2303094ee1c57ba856d246/src/paths.md

## Additions to the documentation syntax

1. `[Iterator](std::iter::Iterator)`
2. `[Iterator][iter]`,
  and somewhere else in the document: `[iter]: std::iter::Iterator`
3. `<std::iter::Iterator>`

## How it will be rendered

The following:

```rust
The offers several ways to fooify [Bars](bars::Bar).
```

should be rendered as:

```html
The offers several ways to fooify <a href="bars/struct.Bar.html">Bars</a>.
```

when on the crates index page (`index.html`),
and as this
when on the page for the `foos` module (`foos/index.html`):

```html
The offers several ways to fooify <a href="../bars/struct.Bar.html">Bars</a>.
```

## Inline paths

When using the third link syntax (`<URL>`),
a link like `<bars::Bar>`
will be converted to ```[`bars::Bar`](bars::Bar)```.

This means that

```
You can use <bars::Bar> for that.
```

will be rendered as

```html
You can use <a href="../bars/struct.Bar.html"><code>bars::Bar</code></a> for that.
```

## Linking to methods

To link to methods, it may be necessary to use fully-qualified paths,
like `<Foo as Bar>::bar`.
We have yet to analyse in which cases this is necessary,
and this syntax is currently not described in [the reference's section on paths][ref-paths].

## Errors

Ideally, Rustdoc would be able to recognize Rust path syntax,
and if the path cannot be resolved,
print a warning (or an error).

## Complex example

(Excerpt from Diesel's [`expression`][diesel-expression] module.)

[diesel-expression]: https://github.com/diesel-rs/diesel/blob/1daf2581919d82b80c18f00957e5c3d35375c4c0/diesel/src/expression/mod.rs

```rust
// diesel/src/expression/mod.rs

//! AST types representing various typed SQL expressions. Almost all types
//! implement either [`Expression`] or [`AsExpression`].
//!
//! [`Expression`]: Expression
//! [`AsExpression`]: AsExpression

/// Represents a typed fragment of SQL. Apps should not need to implement this
/// type directly, but it may be common to use this as type boundaries.
/// Libraries should consider using <infix_predicate!> or <postfix_predicate!>
/// instead of implementing this directly.
pub trait Expression {
    type SqlType;
}

/// Describes how a type can be represented as an expression for a given type.
/// These types couldn't just implement [`Expression`] directly, as many things
/// can be used as an expression of multiple types. (<String> for example, can
/// be used as either [`VarChar`] or [`Text`]).
///
/// [`Expression`]: Expression
/// [`VarChar`]: diesel::types::VarChar
/// [`Text`]: diesel::types::Text
pub trait AsExpression<T> {
    type Expression: Expression<SqlType=T>;
    fn as_expression(self) -> Self::Expression;
}
```

Please note:

- This uses reference-style Markdown links most often.
  This is considered a good practice is official Rust documentation,
  as it is easier to read than having long URLs in the same line as regular text.
- Even though inline links could be used throughout, we only used them for
  the macro links
  and the link to the `String` type (from `std`, available via the prelude).
  These are all items whose path is either obvious
  (from a documentation perspective, macros currently always live in the crate's root)
  or well known
  (the `String` type is used in most Rust code bases).


# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Extend the documentation chapter of the book with a subchapter on How to Link to Items.
Reference the chapter on the module system.
Maybe present an example use case of a module whose documentation links to several related items.


# Drawbacks
[drawbacks]: #drawbacks

- Rustdoc gets more complex.
- These links won't work when the doc comments are rendered with a default Markdown renderer.
- The Rust paths might conflict with other valid links,
  though we could not think of any.

# Alternatives
[alternatives]: #alternatives

## Syntax alternatives

Introduce special syntax for this:

- [javadoc] and [jsdoc]
  use `{@link java.awt.Panel}`
  or `[link text]{@link namepathOrURL}`

[javadoc]: http://docs.oracle.com/javase/8/docs/technotes/tools/windows/javadoc.html
[jsdoc]: http://usejsdoc.org/tags-inline-link.html


# Unresolved questions
[unresolved]: #unresolved-questions

- Is it possible for Rustdoc to resolve paths?
  Is it easy to implement this?
- There is talk about switching Rustdoc to a different markdown renderer ([pulldown-cmark]).
  Does it support this?
  Does the current renderer?

[pulldown-cmark]: https://github.com/google/pulldown-cmark/
