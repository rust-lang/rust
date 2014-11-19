- Start Date: 2014-11-05
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

## Crate scope for macros

In this document, the "syntax environment" refers to the set of syntax
extensions that can be invoked at a given position in the crate.  The names in
the syntax environment are simple unqualified identifiers such as `panic` and
`vec`.  Informally we may write `vec!` to distinguish from an ordinary item.
However, the exclamation point is really part of the invocation syntax, not the
name, and some syntax extensions are invoked with no exclamation point, for
example item decorators like `deriving`.

Imported macros will not automatically end up in the syntax environment.
Instead, you can bring an imported macro into the syntax environment by
providing the crate ident and a macro name (or wildcard) in a `use macro` view
item:

```rust
use macro std::vec;
use macro std::panic as fail;
use macro core::*;
```

These paths must have exactly two components: the crate ident, and a macro name
or `*`.

The syntax environment still consists of unqualified names.  There's no way to
invoke a macro through a qualified name. This obviates the need to change the
parsing of expressions, patterns, etc.

`macro` is a new keyword.  This is an important part of the proposal, because
it signals that we're deviating from usual name resolution.  `use macro` is a
memorable and searchable name for the feature.

The `use macro` view item only affects the syntax environment of the block or
module where it appears, and only from the point of appearance onward.  Unlike
a normal `use` item, this includes child modules (in the same file or others).

Many macros expand using other "private macros" as an implementation detail.
For example, librustc's `declare_lint!` uses `lint_initializer!`.  The client
should not know about this macro, although it still needs to be exported for
cross-crate use.  For this reason we allow `use macro` within a macro
definition.

```rust
/// Not to be imported directly.
extern macro_rules! lint_initializer { ... }

/// Declare a lint.
extern macro_rules! declare_lint {
    use macro $crate::lint_initializer;

    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static $crate::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    )
}
```

The macro `lint_initializer!` will be visible only during further expansion of
the result of invoking `declare_lint!`.

Procedural macros need their own way to manipulate the syntax environment, but
that's an unstable internal API, so it's outside the scope of this RFC.

# New syntax

We also clean up macro syntax in a way that complements the semantic changes above.

## Macro definition syntax

`macro_rules!` already allows `{ }` for the macro body, but the convention is
`( )` for some reason.  In accepting this RFC we would change to a `{ }`
convention for consistency with the rest of the language.

The new macro can be used immediately. There is no way to `use macro` a macro
defined in the same crate, and no need to do so.

A macro with a `pub` qualifier, i.e.

```rust
pub macro_rules! foo { ... }
```

escapes the syntax environment for the enclosing block/module and becomes
available throughout the rest of the crate (according to depth-first search).
This is like putting `#[macro_escape]` on the module and all its ancestors, but
applies *only* to the macro with `pub`.

## Macro export and re-export

A macro definition qualified by `extern` becomes available to other crates.
That is, it can be the target of `use macro`.  Or put another way,
`extern macro_rules!` works the way `#[macro_export] macro_rules!` does today.
Adding `extern` has no effect on the syntax environment for the current crate.

`pub` and `extern` may be used together on the same macro definition, since
their effects are independent.

We can also re-export macros that were imported from another crate.  This is
accomplished with `extern use macro`.

For example, libcollections defines a `vec!` macro, which would now look like:

```rust
extern macro_rules! vec {
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
extern use macro collections::vec;
```

as long as the module `std::vec` is interface-compatible with
`collections::vec`.

(Actually the current libstd `vec!` is completely different for efficiency, but
it's just an example.)

Because macros are exported in crate metadata as strings, macro re-export "just
works" as soon as `$crate` is available.  It's implemented as part of the
`$crate` branch mentioned above.

## `#[plugin]` attribute

Since macros no longer automatically pollute the syntax environment, we can
load them from every `extern crate`.  (Probably we should exclude
`extern crate`s that aren't at the crate root, because there's no way `$crate`
paths will be correct.)

`#[phase(plugin)]` becomes simply `#[plugin]` and is still feature-gated.  It
only controls whether to search for and run a plugin registrar function.  The
plugin itself will decide whether it's to be linked at runtime, by calling a
`Registry` method.

# Other improvements

These are somewhat related to the above, but could be spun off into separate
RFCs.

## Arguments to `#[plugin]`

`#[plugin]` will take an optional "arguments list" of the form

```rust
#[plugin(foo="bar", ... any metas ...)]
extern crate myplugin;
```

rustc itself will not interpret these attribute [meta
items](http://doc.rust-lang.org/syntax/ast/enum.MetaItem_.html), but will make
them available to the plugin through a `Registry` method.

This facilitates plugin configuration.  The alternative in many cases is to use
interacting side effects between procedural macros, which are harder to reason
about.

# Miscellaneous remarks

In a future where macros are scoped the same way as other items,

```rust
use macro std::vec;
```

would become a deprecated synonym for

```rust
use std::vec!;
```

Maintaining this synonym does not seem like a large burden.

We can add a lint to warn about cases where an `extern` macro has paths that
are not absolute-with-crate or `$crate`-relative.  This will have some
(hopefully rare) false positives, and is not fully fleshed out yet.

# Implementation and transition

I will coordinate implementation of this RFC, and I expect to write most of the
code myself.

Some of the syntax cleanups could be deferred until after 1.0.  However the
semantic changes are enough that many users will re-examine and perhaps edit a
large fraction of their macro definitions.  My thinking is that the cleaned-up
syntax should be available when this happens rather than requiring a separate
pass a few months later.  Also I would really like the first Rust release to
put its best foot forward with macros, not just in terms of semantics but
with a polished and pleasant user experience.

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

Should we forbid `$crate` in non-`extern` macros?  It seems useless, however I
think we should allow it anyway, to encourage the habit of writing `$crate::`
for any references to the local crate.

Should we allow `pub use macro`, which would escape the enclosing block/module
the way `pub macro_rules!` does?

Should we require that `extern use macro` can only appear in the crate root?
It doesn't make a lot of sense to put it elsewhere.

# Acknowledgements

This proposal is edited by Keegan McAllister.  It has been refined through many
engaging discussions with:

* Brian Anderson, Shachaf Ben-Kiki, Lars Bergstrom, Nick Cameron, John Clements, Alex Crichton, Cathy Douglass, Steven Fackler, Manish Goregaokar, Dave Herman, Steve Klabnik, Felix S. Klock II, Niko Matsakis, Matthew McPherrin, Paul Stansifer, Sam Tobin-Hochstadt, Aaron Turon, Huon Wilson, Brendan Zabarauskas, Cameron Zwarich
* *GitHub*: `@bill-myers` `@blaenk` `@comex` `@glaebhoerl` `@Kimundi` `@mitchmindtree` `@mitsuhiko` `@P1Start` `@petrochenkov` `@skinner`
* *Reddit*: `ippa` `Mystor` `Quxxy` `rime-frost` `Sinistersnare` `tejp`
* *IRC*: `bstrie` `ChrisMorgan` `cmr` `Earnestly` `eddyb` `tiffany`

My apologies if I've forgotten you, used an un-preferred name, or accidentally
categorized you as several different people.  Pull requests are welcome :)
