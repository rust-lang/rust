- Feature Name: `intra_rustdoc_links`
- Start Date: 2017-03-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/1946
- Rust Issue: https://github.com/rust-lang/rust/issues/43466

# Summary
[summary]: #summary

Add a notation how to create relative links in documentation comments
(based on Rust item paths)
and extend Rustdoc to automatically turn this into working links.


# Motivation
[motivation]: #motivation

It is good practice in the Rust community to
add documentation to all public items of a crate,
as the API documentation as rendered by Rustdoc is the main documentation of most libraries.
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
  (it has change before, cf. [Rust issue #35236]),
  all manually created links need to be updated.

[Rust issue #35236]: https://github.com/rust-lang/rust/pull/35236

To solve this dilemma,
we propose extending Rustdoc
to be able to generate relative links that work in all contexts.


# Detailed Design
[design]: #detailed-design

[Markdown][md]/[CommonMark] allow writing links in several forms
(the names are from the [CommonMark spec][cm-spec] in version 0.27):

[md]: https://daringfireball.net/projects/markdown/syntax
[CommonMark]: http://commonmark.org
[cm-spec]: http://spec.commonmark.org/0.27/

1. `[link text](URL)`
  ([inline link][il])
2. `[link text][link label]`
  ([reference link][rl],
  link label can also be omitted, cf. [shortcut reference links][srl])
  and somewhere else in the document: `[link label]: URL`
  (this part is called [link reference definition][lrd])
3. `<URL>` which will be turned into the equivalent of `[URL](URL)`
  ([autolink][al], required to start with a schema)

[il]: http://spec.commonmark.org/0.27/#inline-link
[rl]: http://spec.commonmark.org/0.27/#reference-link
[srl]: http://spec.commonmark.org/0.27/#shortcut-reference-link
[al]: http://spec.commonmark.org/0.27/#autolinks
[lrd]: http://spec.commonmark.org/0.27/#link-reference-definitions

We propose that
in each occurrence of `URL`
of inline links and link reference definitions,
it should also be possible to write a Rust path
(as defined [in the reference][ref-paths]).
Additionally, automatic [link reference definitions][lrd] should be generated
to allow easy linking to obvious targets.

[ref-paths]: https://github.com/rust-lang-nursery/reference/blob/2d23ea601f017c106a2303094ee1c57ba856d246/src/paths.md

## Additions To The Documentation Syntax

Rust paths as URLs in inline and reference links:

1. `[Iterator](std::iter::Iterator)`
2. `[Iterator][iter]`,
  and somewhere else in the document: `[iter]: std::iter::Iterator`
3. `[Iterator]`,
  and somewhere else in the document: `[Iterator]: std::iter::Iterator`

## Implied Shortcut Reference Links
[isrl]: #implied-shortcut-reference-links

The third syntax example above shows a
[shortcut reference link][srl],
which is a reference link
whose link text and link label are the same,
and there exists a link reference definition for that label.
For example: `[HashMap]` will be rendered as a link
given a link reference definition like ```[HashMap]: std::collections::HashMap```.

To make linking to items easier,
we introduce "implied link reference definitions":

1. `[std::iter::Iterator]`,
  without having a link reference definition for `Iterator` anywhere else in the document
2. ```[`std::iter::Iterator`]```,
  without having a link reference definition for `Iterator` anywhere else in the document
  (same as previous style but with back ticks to format link as inline code)

If Rustdoc finds a shortcut reference link

1. without a matching link reference definition
2. whose link label,
  after stripping leading and trailing back ticks,
  is a valid Rust path

it will add a link reference definition
for this link label pointing to the Rust path.

[Collapsed reference links][crf] (`[link label][]`) are handled analogously.

[crf]: http://spec.commonmark.org/0.27/#collapsed-reference-link

(This was one of the first ideas suggested
by [CommonMark forum] members
as well as by [Guillaume Gomez].)

[CommonMark forum]: https://talk.commonmark.org/t/what-should-the-rust-community-do-for-linkage/2141
[Guillaume Gomez]: https://github.com/GuillaumeGomez

## Standard-conforming Markdown

These additions are valid Markdown,
as defined by the original [Markdown syntax definition][md]
as well as the [CommonMark] project.
Especially, Rust paths are valid CommonMark [link destinations],
even with the suffixes described [below][path-ambiguities].

[link destinations]: http://spec.commonmark.org/0.27/#link-destination

## How Links Will Be Rendered

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

## No Autolinks Style

When using the autolink syntax (`<URL>`),
the URL has to be an [absolute URI],
i.e., it has to start with an URI scheme.
Thus, it will not be possible to write `<Foo>`
to link to a Rust item called `Foo`
that is in scope
(this also conflicts with Markdown ability to contain arbitrary HTML elements).
And while `<std::iter::Iterator>` is a valid URI
(treating `std:` as the scheme),
to avoid confusion, the RFC does not propose adding any support for autolinks.

[absolute URI]: http://spec.commonmark.org/0.27/#absolute-uri

This means that this **will not** render a valid link:

```markdown
Does not work: <bars::Bar> :(
```

It will just output what any CommonMark compliant renderer would generate:

```html
Does not work: <a href="bars::Bar">bars::Bar</a> :(
```

We suggest to use [Implied Shortcut Reference Links][isrl] instead:

```markdown
Does work: [`bars::Bar`] :)
```

which will be rendered as

```html
Does work: <a href="../bars/struct.Bar.html"><code>bars::Bar</code></a> :)
```

## Resolving Paths

The Rust paths used in links are resolved
relative to the item in whose documentation they appear.
Specifically, when using inner doc comments (`//!`, `/*!`),
the paths are resolved from the inside of the item,
while regular doc comments (`///`, `/**`) start from the parent scope.

Here's an example:

```rust
/// Container for a [Dolor](ipsum::Dolor).
struct Lorem(ipsum::Dolor);

/// Contains various things, mostly [Dolor](ipsum::Dolor) and a helper function,
/// [sit](ipsum::sit).
mod ipsum {
    pub struct Dolor;

    /// Takes a [Dolor] and does things.
    pub fn sit(d: Dolor) {}
}

mod amet {
  //! Helper types, can be used with the [ipsum](super::ipsum) module.
}
```

And here's an edge case:

```rust
use foo::Iterator;

/// Uses `[Iterator]`. <- This resolves to `foo::Iterator` because it starts
/// at the same scope as `foo1`.
fn foo1() { }

fn foo2() {
    //! Uses `[Iterator]`. <- This resolves to `bar::Iterator` because it starts
    //! with the inner scope of `foo2`'s body.

    use bar::Iterator;
}
```

## Path Ambiguities
[path-ambiguities]: #path-ambiguities

Rust has three different namespaces that items can be in,
types, values, and macros.
That means that in a given source file,
three items with the same name can be used,
as long as they are in different namespaces.

To illustrate, in the following example
we introduce an item called `FOO` in each namespace:

```rust
pub trait FOO {}

pub const FOO: i32 = 42;

macro_rules! FOO { () => () }
```

To be able to link to each item,
we'll need a way to disambiguate the namespaces.
Our proposal is this:

- Links to types are written as described earlier,
  with no pre- or suffix,
  e.g., `Look at the [FOO] trait`.
  For consistency,
  it is also possible to prefix the type with the concrete item type:
  - Links to `struct`s can be prefixed with `struct `,
    e.g., `See [struct Foo]`.
  - Links to `enum`s can be prefixed with `enum `,
    e.g., `See [enum foo]`.
  - Links to type aliases can be prefixed with `type `,
    e.g., `See [type foo]`.
  - Links to modules can be prefixed with `mod `,
    e.g., `See [mod foo]`.
- In links to macros,
  the link label must end with a `!`,
  e.g., `Look at the [FOO!] macro`.
- For links to values, we differentiate three cases:
  - Links to functions are written with a `()` suffix,
    e.g., `Also see the [foo()] function`.
  - Links to constants are prefixed with `const `,
    e.g., `As defined in [const FOO].`
  - Links to statics are prefixed with `static `,
    e.g., `See [static FOO]`.

It should be noted that in the RFC discussion it was determined
that exact knowledge of the item type
should not be necessary; only knowing the namespace should suffice.
It is acceptable that the tool resolving the links
allows (and successfully resolves) a link
with the wrong prefix that is in the same namespace.
E.g., given an `struct Foo`, it may be possible to link to it using `[enum Foo]`,
or, given a `mod bar`, it may be possible to link to that using `[struct bar]`.


## Errors
[errors]: #errors

Ideally, Rustdoc would be able to recognize Rust path syntax,
and if the path cannot be resolved,
print a warning (or an error).
These diagnostic messages should highlight the specific link
that Rustdoc was not able to resolve,
using the original Markdown source from the comment and correct line numbers.

## Complex Example
[complex-example]: #complex-example

(Excerpt from Diesel's [`expression`][diesel-expression] module.)

[diesel-expression]: https://github.com/diesel-rs/diesel/blob/1daf2581919d82b80c18f00957e5c3d35375c4c0/diesel/src/expression/mod.rs

```rust
// diesel/src/expression/mod.rs

//! AST types representing various typed SQL expressions. Almost all types
//! implement either [`Expression`] or [`AsExpression`].

/// Represents a typed fragment of SQL. Apps should not need to implement this
/// type directly, but it may be common to use this as type boundaries.
/// Libraries should consider using [`infix_predicate!`] or
/// [`postfix_predicate!`] instead of implementing this directly.
pub trait Expression {
    type SqlType;
}

/// Describes how a type can be represented as an expression for a given type.
/// These types couldn't just implement [`Expression`] directly, as many things
/// can be used as an expression of multiple types. ([`String`] for example, can
/// be used as either [`VarChar`] or [`Text`]).
///
/// [`VarChar`]: diesel::types::VarChar
/// [`Text`]: diesel::types::Text
pub trait AsExpression<T> {
    type Expression: Expression<SqlType=T>;
    fn as_expression(self) -> Self::Expression;
}
```

Please note:

- This uses implied shortcut reference links most often.
  Since the original documentation put the type/trait names in back ticks to render them as code, we preserved this style.
  (We don't propose this as a general convention, though.)
- Even though implied shortcut reference links could be used throughout,
  they are not used for the last two links (to `VarChar` and `Text`),
  which are not in scope and need to be linked to by their absolute Rust path.
  To make reading easier and less noisy, reference links are used to rename the links.
  (An assumption is that most readers will recognize these names and know they are part of `diesel::types`.)


# How We Teach This
[how-we-teach-this]: #how-we-teach-this

- Extend the documentation chapter of the book with a subchapter on How to Link to Items.
- Reference the chapter on the module system, to let reads familiarize themselves with Rust paths.
- Maybe present an example use case of a module whose documentation links to several related items.


# Drawbacks
[drawbacks]: #drawbacks

- Rustdoc gets more complex.
- These links won't work when the doc comments are rendered with a default Markdown renderer.
- The Rust paths might conflict with other valid links,
  though we could not think of any.


# Possible Extensions
[possible-extensions]: #possible-extensions

## Linking to Fields

To link to the fields of a `struct`
we propose to write the path to the struct,
followed by a dot, followed by the field name.

For example:

```markdown
This is stored in the [`size`](storage::Filesystem.size) field.
```

## Linking to Enum Variants

To link to the variants of an `enum`,
we propose to write the path to the enum,
followed by two colons, followed by the field name,
just like `use Foo::Bar` can be used to import the `Bar` variant of an `enum Foo`.

For example:

```markdown
For custom settings, supply the [`Custom`](storage::Engine::Other) field.
```

## Linking to associated Items

To link to associated items,
i.e., the associated functions, types, and constants of a trait,
we propose to write the path to the trait,
followed by two colons, followed by the associated item's name.
It may be necessary to use fully-qualified paths
(cf. [the reference's section on disambiguating function calls][ref-ufcs]),
like `See the [<Foo as Bar>::bar()] method`.
We have yet to analyze in which cases this is necessary,
and what syntax should be used.

[ref-ufcs]: https://github.com/rust-lang-nursery/reference/blob/96e976d32a0a6927dd26c2ee805aaf44ef3bef2d/src/expressions.md#disambiguating-function-calls

## Linking to External Documentation

Currently, Rustdoc is able to link to external crates,
and renders documentation for all dependencies by default.
Referencing the standard library (or `core`)
generates links with a well-known base path,
e.g. `https://doc.rust-lang.org/nightly/`.
Referencing other external crates
links to the pages Rustdoc has already rendered (or will render) for them.
Special flags (e.g. `cargo doc --no-deps`) will not change this behavior.

We propose to generalize this approach
by adding parameters to rustdoc
that allow overwriting the base URLs
it used for external crate links.
(These parameters will at first
be supplied as CLI flags
but could also be given via a config file,
environment variables,
or other means in the future.)

We suggest the following syntax:

```sh
rustdoc --extern-base-url="regex=https://docs.rs/regex/0.2.2/regex/" [...]
```

By default, the core/std libraries should have a default base URL
set to the latest known Rust release when the version of rustdoc was built.

In addition to that,
`cargo doc` _may_ be extended with CLI flags
to allow shortcuts to some common usages.
E.g., a `--external-docs` flag may add base URLs using [docs.rs]
for all crates that are from the crates.io repository
(docs.rs automatically renders documentation for crates published to crates.io).

[docs.rs]: https://docs.rs/

### Known Issues

Automatically linking to external docs has the following known tradeoffs:

- The generated URLs may not/no longer exist
  - Not all crate documentation can be rendered without a known local setup,
    e.g., for crates that use procedural macros/build scripts
    to generate code based on the local environment.
  - Not all crate documentation can be rendered without having  3rd-party tools installed.
- The generated URLs may not/no have the expected content, because
  - The exact Cargo features used to build a crate locally
    were not used when building the docs available at the given URL.
  - The crate has platform-specific items,
    and the local platform and the platform
    used to render the docs available at the given URL
    differ
    (note that docs.rs renders docs for multiple platforms, though).

# Alternatives
[alternatives]: #alternatives

- Prefix Rust paths with a URI scheme, e.g. `rust:`
  (cf. [path ambiguities][path-ambiguities]).
- Prefix Rust paths with a URI scheme for the item type, e.g. `struct:`, `enum:`, `trait:`, or `fn:`.

- [javadoc] and [jsdoc]
  use `{@link java.awt.Panel}`
  or `[link text]{@link namepathOrURL}`

[javadoc]: http://docs.oracle.com/javase/8/docs/technotes/tools/windows/javadoc.html
[jsdoc]: http://usejsdoc.org/tags-inline-link.html

- [@kennytm](https://github.com/kennytm)
  listed other syntax alternatives
  [here](https://github.com/rust-lang/rfcs/pull/1946#issuecomment-284718018).


# Unresolved Questions
[unresolved]: #unresolved-questions

- Is it possible for Rustdoc to resolve paths?
  Is it easy to implement this?
- There is talk about switching Rustdoc to a different markdown renderer ([pulldown-cmark]).
  Does it support this?
  Does the current renderer?

[pulldown-cmark]: https://github.com/google/pulldown-cmark/
