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
- It's confusing that macro definition is itself a macro invocation, with a side effect
  on the expansion context

These issues in particular are things we have a chance of addressing for 1.0.
This RFC contains plans to do so.

# Detailed design

Skip ahead to

* [`macro` items](#macro-items)
* [`$crate`](#crate)
* [Crate scope for macros](#crate-scope-for-macros)
* [Item macro sugar](#item-macro-sugar)
* [Macro re-export](#macro-re-export)
* [`#[plugin]` attribute](#plugin-attribute)
* [Unspecify order of procedural macro side effects](#unspecify-order-of-procedural-macro-side-effects)

## `macro` items

Introduce a new keyword `macro`.  Use it for a new kind of item, a macro definition:

```rust
// first example from the Macros Guide
macro early_return {
    ($inp:expr $sp:ident) => (
        match $inp {
            $sp(x) => { return x; }
            _ => {}
        }
    )
}
```

`macro_rules!` already allows `{ }` for the macro body, but the convention is
`( )` for some reason.  In accepting this RFC we would change to a `{ }`
convention for consistency with the rest of the language.

There are two scope qualifiers for `macro`:

* `pub` — this macro "escapes" up the module hierarchy to the crate root, so it
  can be used anywhere in this crate after its definition (according to a
  depth-first traversal).  This is like putting `#[macro_escape]` on the module
  and all its ancestors, but applies *only* to the macro with the `#[visible]`
  attribute.

* `extern` — this macro can be imported by other crates, i.e. the same meaning
  as `#[macro_export]` today

These can be used together.

The default (as today) is that the macro is only visible within the lexical
scope where it is defined.

The `Item` AST node changes as follows:

```rust
pub enum Item_ {
    // ...

    /// A macro definition.
    ItemDefineMacro(Vec<TokenTree>),

    /// A macro invocation.
    ItemUseMacro(Mac),  // was called ItemMac
}
```

While it's unfortunate that AST types can't represent a `macro` item as
anything richer than a token tree, this is not a regression from the status quo
with `macro_rules!`, which parses as an `ItemMac`.

This also provides flexibility to make backwards-compatible changes.  One can
imagine

```rust
#[procedural] macro match_token {
    fn expand(cx: &mut ExtCtxt, span: Span, toks: &[TokenTree])
            -> Box<MacResult+'static> {
        // ...
    }
}
```

or

```rust
macro atom {
    ($name:tt) => fn expand(...) {
        // ...
    }
}
```

though working out the details is far outside the scope of this RFC.

We are free to change the AST and parsing after 1.0 as long as the old syntax
still works.  So we could have a separate enum variant for procedural macros
parsed as function decls.

## `$crate`

Add a special metavar `$crate` which expands to `::foo` when the macro was
imported from crate `foo`, and to nothing when it was defined in-crate.
`$crate::bar::baz` will be an absolute path either way.

This feature eliminates the need for the "curious inner-module" and also
enables macro re-export (see below).  It is [implemented and
tested](https://github.com/kmcallister/rust/commits/macro-reexport) but needs a
rebase.

Add a lint to warn in cases where an `extern` macro has paths that are not
absolute-with-crate or `$crate`-relative.  This will have some (hopefully rare)
false positives, and is not fully fleshed out yet.

## Crate scope for macros

Instead of a single global namespace for macro definitions, we now have one
namespace per crate.  We introduce a new view item, `use macro`:

```rust
use macro std::vec;
use macro std::panic as fail;
```

There's no way to invoke a macro with a qualified name; this obviates the need
to change the parsing of expressions, patterns, etc.

The `macro` keyword is important because it signals that this is something
different from the usual name resolution.  `use macro` is a memorable and
searchable name for the feature.

`use macro` can be qualified with `pub` to get the same "escape" behavior as
`pub macro`.  `use macro` also allows globs.  For example:

```rust
#![no_std]
extern crate core;
pub use macro core::*;
```

Many macros expand using other "private macros" as an implementation detail.
For example, librustc's `declare_lint!` uses `lint_initializer!`.  The client
should not know about this macro, although it still needs to be exported for
cross-crate use.  For this reason we allow `use macro` within a macro
definition, and allow `$crate` in that context.

```rust
/// Not to be imported directly.
extern macro lint_initializer { ... }

/// Declare a lint.
extern macro declare_lint {
    use macro $crate::lint_initializer;

    ($name:ident, $level:ident, $desc:expr) => (
        static $name: &'static $crate::lint::Lint
            = &lint_initializer!($name, $level, $desc);
    )
}
```

The macro `lint_initializer!` will be visible only during further expansion of
the result of invoking `declare_lint!`.

Procedural macros need their own way to manipulate the expansion context, but
that's an unstable internal API, so it's outside the scope of this RFC.

In the long run,

```rust
use macro std::vec;
```

may end up as a deprecated synonym for

```rust
use std::vec!;
```

but maintaining this synonym does not seem like a large burden.

## Item macro sugar

An item defines one name in the current module, and can have "adjective"
qualifiers such as `pub`, `extern`, etc.  We extend macro invocation in item
position to reflect this form.  The new invocation syntax is

```rust
<quals...> foo! <ident> {
    <body...>
}
```

where `<quals...>` is a sequence of zero or more keywords from the set `pub`
`priv` `extern` `unsafe` `const` `static` `box` `ref` `mut`.  This list is
pretty arbitrary but can be expanded later.  For now it only includes keywords
that have existing meaning(s) in Rust, which should keep things somewhat under
control.

This form of item macro is a bit like defining your own keyword that can take
the place of `fn`, `struct`, etc.

We keep the existing syntax for invoking macros in item position, and desugar
the above syntax to it, as

```rust
foo!(<quals...> : <ident> { <body...> })
```

This extends the existing `IdentTT` macro form and makes it available to
non-procedural macros, while reducing special cases within the compiler.

An item macro invocation in either the sugared or unsugared form may expand to
zero or more items.

Item macro sugar is an integral part of macro reform because it's needed to
make `pub` and `extern` work with macro-defining macros.  For example

```rust
macro mega_macro {
    ($($qual:ident)* : $name:ident $body:tt) => (
        $($qual)* macro $name {
            // ...
        }
    )
}
```

can be invoked as

```rust
pub extern mega_macro! foo {
    (bar) => (3u)
}
```

Here `mega_macro!` takes the place of the built-in `macro` keyword.

The applications go beyond macro-defining macros.  For example, [this macro in
rustc](https://github.com/rust-lang/rust/blob/221fc1e3cdcc208e1bb7debcc2de27d47c847747/src/librustc/lint/mod.rs#L83-L95)
could change to support invocations like

```rust
lint! UNUSED_ATTRIBUTES {
    Warn, "detects attributes that were not used by the compiler"
}

pub lint! PATH_STATEMENTS {
    Warn, "path statements with no effect"
}
```

which is considerably nicer than the current syntax.
[`lazy-static.rs`](https://github.com/Kimundi/lazy-static.rs) could support
syntax like

```rust
pub lazy_static! LOG {
   : Mutex<Vec<Event>>
   = Mutex::new(Vec::with_capacity(50_000))
}
```

## Macro re-export

With `$crate` we can easily re-export macros that were imported from another
crate.  This is accomplished with `extern use macro`.

For example, libcollections defines a `vec!` macro, which would now look like:

```rust
extern macro vec {
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

## `#[plugin]` attribute

Since macros are now crate-scoped, we can load macros from every `extern crate`
without a special attribute.  (Probably we should exclude `extern crate`s that
aren't at the crate root, because there's no way `$crate` paths will be
correct.)

`#[phase(plugin)]` becomes simply `#[plugin]` and is still feature-gated.  It
only controls whether to search for and run a plugin registrar function.  The
plugin itself will decide whether it's to be linked at runtime, by calling a
`Registry` method.

`#[plugin]` takes an optional "arguments list" of the form

```rust
#[plugin(foo="bar", ... any metas ...)]
extern crate myplugin;
```

rustc itself will not interpret these attribute [meta
items](http://doc.rust-lang.org/syntax/ast/enum.MetaItem_.html), but will make
them available to the plugin through a `Registry` method.

## Unspecify order of procedural macro side effects

We clarify that the ordering of expansion, hence side effects, for separate
(i.e. non-nested) procedural macro invocations is unspecified.  This does not
affect the stable language, because procedural macros are not part of it, and
expansion of pattern-based macros cannot have side effects.

Interacting side effects between procedural macros is messy in general.  It's
much better for a macro to approximate a pure function of its input, plus an
"environment" that does not change during macro expansion.  I claim that this
is usually possible (see
[discussion](https://github.com/rust-lang/rfcs/pull/453#issuecomment-62813856)).

The main reason to consider this now in an RFC is to make sure that built-in
procedural macros exposed to stable code can comply.  Reserving the right to
change expansion order allows us to pursue more sophisticated approaches to the
name resolution problem after 1.0.

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

# Acknowledgements

This proposal is edited by Keegan McAllister.  It has been refined through many
engaging discussions with:

* Brian Anderson, Shachaf Ben-Kiki, Lars Bergstrom, Nick Cameron, John Clements, Alex Crichton, Cathy Douglass, Steven Fackler, Manish Goregaokar, Dave Herman, Steve Klabnik, Felix S. Klock II, Niko Matsakis, Matthew McPherrin, Paul Stansifer, Sam Tobin-Hochstadt, Aaron Turon, Huon Wilson, Brendan Zabarauskas, Cameron Zwarich
* *GitHub*: `@bill-myers` `@blaenk` `@comex` `@glaebhoerl` `@Kimundi` `@mitchmindtree` `@mitsuhiko` `@P1Start` `@petrochenkov` `@skinner`
* *Reddit*: `ippa` `Mystor` `Quxxy` `rime-frost` `Sinistersnare` `tejp`
* *IRC*: `bstrie` `ChrisMorgan` `cmr` `Earnestly` `eddyb` `tiffany`

My apologies if I've forgotten you, used an un-preferred name, or accidentally
categorized you as several different people.  Pull requests are welcome :)
