- Start Date: 2014-11-05
- RFC PR: [rust-lang/rfcs#453](https://github.com/rust-lang/rfcs/pull/453)
- Rust Issue: [rust-lang/rust#20008](https://github.com/rust-lang/rust/issues/20008)

# Summary

Various enhancements to macros ahead of their standardization in 1.0.

**Note**: This is not the final Rust macro system design for all time.  Rather,
it addresses the largest usability problems within the limited time frame for
1.0.  It's my hope that a lot of these problems can be solved in nicer ways
in the long term (there is some discussion of this below).

# Motivation

`macro_rules!` has [many rough
edges](https://github.com/rust-lang/rfcs/issues/440).  A few of the big ones:

- You can't re-export macros
- Even if you could, names produced by the re-exported macro won't follow the re-export
- You can't use the same macro in-crate and exported, without the "curious inner-module" hack
- There's no namespacing at all
- You can't control which macros are imported from a crate
- You need the feature-gated `#[phase(plugin)]` to import macros

These issues in particular are things we have a chance of addressing for 1.0.
This RFC contains plans to do so.

# Semantic changes

These are the substantial changes to the macro system.  The examples also use
the improved syntax, described later.

## `$crate`

The first change is to disallow importing macros from an `extern crate` that is
not at the crate root.  In that case, if

```rust
extern crate "bar" as foo;
```

imports macros, then it's also introducing ordinary paths of the form
`::foo::...`.  We call `foo` the *crate ident* of the `extern crate`.

We introduce a special macro metavar `$crate` which expands to `::foo` when a
macro was imported through crate ident `foo`, and to nothing when it was
defined in the crate where it is being expanded.  `$crate::bar::baz` will be an
absolute path either way.

This feature eliminates the need for the "curious inner-module" and also
enables macro re-export (see below).  It is [implemented and
tested](https://github.com/kmcallister/rust/commits/macro-reexport) but needs a
rebase.

We can add a lint to warn about cases where an exported macro has paths that
are not absolute-with-crate or `$crate`-relative.  This will have some
(hopefully rare) false positives.

## Macro scope

In this document, the "syntax environment" refers to the set of syntax
extensions that can be invoked at a given position in the crate.  The names in
the syntax environment are simple unqualified identifiers such as `panic` and
`vec`.  Informally we may write `vec!` to distinguish from an ordinary item.
However, the exclamation point is really part of the invocation syntax, not the
name, and some syntax extensions are invoked with no exclamation point, for
example item decorators like `deriving`.

We introduce an attribute `macro_use` to specify which macros from an external
crate should be imported to the syntax environment:

```rust
#[macro_use(vec, panic="fail")]
extern crate std;

#[macro_use]
extern crate core;
```

The list of macros to import is optional. Omitting the list imports all macros,
similar to a glob `use`.  (This is also the mechanism by which `std` will
inject its macros into every non-`no_std` crate.)

Importing with rename is an optional part of this proposal that will be
implemented for 1.0 only if time permits.

Macros imported this way can be used anywhere in the module after the
`extern crate` item, including in child modules.  Since a macro-importing
`extern crate` must appear at the crate root, and view items come before
other items, this effectively means imported macros will be visible for
the entire crate.

Any name collision between macros, whether imported or defined in-crate, is a
hard error.

Many macros expand using other "helper macros" as an implementation detail.
For example, librustc's `declare_lint!` uses `lint_initializer!`.  The client
should not know about this macro, although it still needs to be exported for
cross-crate use.  For this reason we allow `#[macro_use]` on a macro
definition.

```rust
/// Not to be imported directly.
#[macro_export]
macro_rules! lint_initializer { ... }

/// Declare a lint.
#[macro_export]
#[macro_use(lint_initializer)]
macro_rules! declare_lint {
    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static $crate::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    )
}
```

The macro `lint_initializer!`, imported from the same crate as `declare_lint!`,
will be visible only during further expansion of the result of invoking
`declare_lint!`.

`macro_use` on `macro_rules` is an optional part of this proposal that will be
implemented for 1.0 only if time permits.  Without it, libraries that use
helper macros will need to list them in documentation so that users can import
them.

Procedural macros need their own way to manipulate the syntax environment, but
that's an unstable internal API, so it's outside the scope of this RFC.

# New syntax

We also clean up macro syntax in a way that complements the semantic changes above.

## `#[macro_use(...)] mod`

The `macro_use` attribute can be applied to a `mod` item as well.  The
specified macros will "escape" the module and become visible throughout the
rest of the enclosing module, including any child modules.  A crate might start
with

```rust
#[macro_use]
mod macros;
```

to define some macros for use by the whole crate, without putting those
definitions in `lib.rs`.

Note that `#[macro_use]` (without a list of names) is equivalent to the
current `#[macro_escape]`.  However, the new convention is to use an outer
attribute, in the file whose syntax environment is affected, rather than an
inner attribute in the file defining the macros.

## Macro export and re-export

Currently in Rust, a macro definition qualified by `#[macro_export]` becomes
available to other crates.  We keep this behavior in the new system.  A macro
qualified by `#[macro_export]` can be the target of `#[macro_use(...)]`, and
will be imported automatically when `#[macro_use]` is given with no list of
names.

`#[macro_export]` has no effect on the syntax environment for the current
crate.

We can also re-export macros that were imported from another crate.  For
example, libcollections defines a `vec!` macro, which would now look like:

```rust
#[macro_export]
macro_rules! vec {
    ($($e:expr),*) => ({
        let mut _temp = $crate::vec::Vec::new();
        $(_temp.push($e);)*
        _temp
    })
}
```

Currently, libstd duplicates this macro in its own `macros.rs`.  Now it could
do

```rust
#[macro_reexport(vec)]
extern crate collections;
```

as long as the module `std::vec` is interface-compatible with
`collections::vec`.

(Actually the current libstd `vec!` is completely different for efficiency, but
it's just an example.)

Because macros are exported in crate metadata as strings, macro re-export "just
works" as soon as `$crate` is available.  It's implemented as part of the
`$crate` branch mentioned above.

## `#[plugin]` attribute

`#[phase(plugin)]` becomes simply `#[plugin]` and is still feature-gated.  It
only controls whether to search for and run a plugin registrar function.  The
plugin itself will decide whether it's to be linked at runtime, by calling a
`Registry` method.

`#[plugin]` can optionally take any [meta
items](http://doc.rust-lang.org/syntax/ast/enum.MetaItem_.html) as "arguments",
e.g.

```rust
#[plugin(foo, bar=3, baz(quux))]
extern crate myplugin;
```

rustc itself will not interpret these arguments, but will make them available
to the plugin through a `Registry` method.  This facilitates plugin
configuration.  The alternative in many cases is to use interacting side
effects between procedural macros, which are harder to reason about.

## Syntax convention

`macro_rules!` already allows `{ }` for the macro body, but the convention is
`( )` for some reason.  In accepting this RFC we would change to a `{ }`
convention for consistency with the rest of the language.

## Reserve `macro` as a keyword

A lot of the syntax alternatives discussed for this RFC involved a `macro`
keyword.  The consensus is that macros are too unfinished to merit using the
keyword now.  However, we should reserve it for a future macro system.

# Implementation and transition

I will coordinate implementation of this RFC, and I expect to write most of the
code myself.

To ease the transition, we can keep the old syntax as a deprecated synonym, to
be removed before 1.0.

# Drawbacks

This is big churn on a major feature, not long before 1.0.

We can ship improved versions of `macro_rules!` in a back-compatible way (in
theory; I would like to smoke test this idea before 1.0).  So we could defer
much of this reform until after 1.0.  The main reason not to is macro
import/export.  Right now every macro you import will be expanded using your
local copy of `macro_rules!`, regardless of what the macro author had in mind.

# Alternatives

We could try to implement proper hygienic capture of crate names in macros.
This would be wonderful, but I don't think we can get it done for 1.0.

We would have to actually parse the macro RHS when it's defined, find all the
paths it wants to emit (somehow), and then turn each crate reference within
such a path into a globally unique thing that will still work when expanded in
another crate.  Right now libsyntax is oblivious to librustc's name resolution
rules, and those rules can't be applied until macro expansion is done, because
(for example) a macro can expand to a `use` item.

nrc suggested dropping the `#![macro_escape]` functionality as part of this
reform.  Two ways this could work out:

- *All* macros are visible throughout the crate.  This seems bad; I depend on
  module scoping to stay (marginally) sane when working with macros.  You can
  have private helper macros in two different modules without worrying that
  the names will clash.

- Only macros at the crate root are visible throughout the crate.  I'm also
  against this because I like keeping `lib.rs` as a declarative description
  of crates, modules, etc. without containing any actual code.  Forcing the
  user's hand as to which file a particular piece of code goes in seems
  un-Rusty.

# Unresolved questions

Should we forbid `$crate` in non-exported macros?  It seems useless, however I
think we should allow it anyway, to encourage the habit of writing `$crate::`
for any references to the local crate.

Should `#[macro_reexport]` support the "glob" behavior of `#[macro_use]` with
no names listed?

# Acknowledgements

This proposal is edited by Keegan McAllister.  It has been refined through many
engaging discussions with:

* Brian Anderson, Shachaf Ben-Kiki, Lars Bergstrom, Nick Cameron, John Clements, Alex Crichton, Cathy Douglass, Steven Fackler, Manish Goregaokar, Dave Herman, Steve Klabnik, Felix S. Klock II, Niko Matsakis, Matthew McPherrin, Paul Stansifer, Sam Tobin-Hochstadt, Erick Tryzelaar, Aaron Turon, Huon Wilson, Brendan Zabarauskas, Cameron Zwarich
* *GitHub*: `@bill-myers` `@blaenk` `@comex` `@glaebhoerl` `@Kimundi` `@mitchmindtree` `@mitsuhiko` `@P1Start` `@petrochenkov` `@skinner`
* *Reddit*: `gnusouth` `ippa` `!kibwen` `Mystor` `Quxxy` `rime-frost` `Sinistersnare` `tejp` `UtherII` `yigal100`
* *IRC*: `bstrie` `ChrisMorgan` `cmr` `Earnestly` `eddyb` `tiffany`

My apologies if I've forgotten you, used an un-preferred name, or accidentally
categorized you as several different people.  Pull requests are welcome :)
