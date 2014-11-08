- Start Date: 2014-11-05
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Various enhancements to `macro_rules!` ahead of its standardization in 1.0.

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

# Detailed design

## Rename

Rename `macro_rules!` to `macro!`, analogous to `fn`, `struct`, etc.

## `#[visible(...)]`

`#[macro_export]` and `#[macro_escape]` are replaced with a new attribute
applied to a `macro!` invocation:

* `#[visible(this_crate)]` — this macro "escapes" up the module hierarchy to
  the crate root, so it can be used anywhere in the crate after the definition
  (according to a depth-first traversal).  This is like putting `#[macro_escape]`
  on the module and all its ancestors, but applies *only* to the macro with the
  `#[visible]` attribute.

* `#[visible(other_crates)]` — same meaning as `#[macro_export]` today

These can be combined as `#[visible(this_crate, other_crates)]` (in either
order).

The default (as today) is that the macro is only visible within the lexical
scope where it is defined.

## `$crate`

Add a special metavar `$crate` which expands to `::foo` when the macro was
imported from crate `foo`, and to nothing when it was defined in-crate.
`$crate::bar::baz` will be an absolute path either way.

This feature eliminates the need for the "curious inner-module" and also
enables macro re-export (see below).  It is [implemented and
tested](https://github.com/kmcallister/rust/commits/macro-reexport) but needs a
rebase.

## Crate scope for macros

Instead of a single global namespace for macro definitions, we now have one
namespace per crate.  We introduce an attribute to `use` macros with optional
renaming:

```rust
#[use_macros(std::vec, std::panic as fail)]
```

(We'd need to extend attribute syntax, or change this to be compatible.)

This can be applied to a module, a function, or anywhere else attributes are
allowed.

There's no way to invoke a macro with a qualified name; this obviates the need
for changes to the grammar / parser.

This isn't an actual `use` item because macro expansion happens before name
resolution, and libsyntax knows nothing about the latter.

This change applies to `macro!` and to all other syntax extensions, even
decorators that are used as attributes.

Many macros expand using other "private macros" as an implementation detail.
For example, librustc's `declare_lint!` uses `lint_initializer!`.  The client
should not know about this macro, although it still needs to be exported for
cross-crate use.  The solution is to allow `use_macros` on `macro!` itself, and
allow `$crate` in that context:

```rust
/// Not to be imported directly.
#[visible(other_crates)]
macro! lint_initializer ( ... )

/// Declare a lint.
#[visible(other_crates)]
#[use_macros($crate::lint_initializer)]
macro! declare_lint (
    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static $crate::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    )
)
```

The macro `lint_initializer!` will be visible only during further expansion of
the result of invoking `declare_lint!`.

Procedural macros need their own way to manipulate the expansion context, but
that's an unstable internal API, so it's outside the scope of this RFC.
Ideally the implementation of `macro!` will use the same API.

## Macro re-export

With `$crate` we can easily re-export macros that were imported from another
crate.

For example, libcollections defines a `vec!` macro, which would now look like:

```rust
#[visible(other_crates)]
macro! vec (
    ($($e:expr),*) => ({
        let mut _temp = $crate::vec::Vec::new();
        $(_temp.push($e);)*
        _temp
    })
)
```

Currently, libstd duplicates this macro in its own `macros.rs`.  Now it could
do

```rust
#![reexport_macros(collections::vec)]
```

as long as the module `std::vec` is interface-compatible with
`collections::vec`.

(Actually the current libstd `vec!` is completely different for efficiency, but
it's just an example.)

## Auto-load macros

Since macros are now crate-scoped, we can load macros from every `extern crate`
without a special attribute.  (Probably we should exclude `extern crate`s that
aren't at the crate root, because there's no way `$crate` paths will be
correct.)

`#[phase(plugin)]` becomes simply `#[plugin]` and is still feature-gated.  It
only controls whether to search for and run a plugin registrar function.  The
plugin itself will decide whether it's to be linked at runtime, by calling a
`Registry` method.

In the future I would like to support `#[plugin(... any metas ...)]` where
these "arguments" are available in the registrar and can be used to configure
how the plugin works.  This RFC does not cover that feature; I just want to
make sure our design is compatible.


# Drawbacks

This is big churn on a major feature, not long before 1.0.

We can ship improved versions of `macro!` in a back-compat way (in theory; I
would like to smoke test this idea before 1.0).  So we could defer all this
reform until after 1.0.  The main reason not to is macro import/export.  Right
now every macro you import will be expanded using your local copy of `macro!`,
regardless of what the macro author had in mind.

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

Do we support globs in `use_macros`?  How does prelude injection work?
How do I get all of libcore's macros if my crate is `no_std`?

Does `use_macros` also define the name in child modules? In a sense this is the
more natural thing to do in libsyntax, but it's inconsistent with normal `use`
items.

All of the syntax is bikeshedable. For example, should `use_macros` include
the exclamation point? What about when it applies to item decorator attributes?

Allowing `$crate` in attributes is weird. Maybe we should use some other
syntax that fits with existing attribute parsing.
