# Module system cleanups

- Start Date: 2014-10-10
- RFC PR: [rust-lang/rfcs#385](https://github.com/rust-lang/rfcs/pull/385)
- Rust Issue: [rust-lang/rust#18219](https://github.com/rust-lang/rust/issues/18219)

# Summary

- Lift the hard ordering restriction between `extern crate`, `use` and other items.
- Allow `pub extern crate` as opposed to only private ones.
- Allow `extern crate` in blocks/functions, and not just in modules.

# Motivation

The main motivation is consistency and simplicity:
None of the changes proposed here change the module system in any meaningful way,
they just remove weird forbidden corner cases that are all already possible to express today with workarounds.

Thus, they make it easier to learn the system for beginners, and easier to for developers to evolve their module hierarchies

## Lifting the ordering restriction between `extern crate`, `use` and other items.

Currently, certain items need to be written in a fixed order: First all `extern crate`, then all `use` and then all other items.
This has historically reasons, due to the older, more complex resolution algorithm, which included that shadowing was allowed between those items in that order,
and usability reasons, as it makes it easy to locate imports and library dependencies.

However, after [RFC 50](https://github.com/rust-lang/rfcs/blob/master/complete/0050-no-module-shadowing.md) got accepted there
is only ever one item name in scope from any given source so the historical "hard" reasons loose validity:
Any resolution algorithm that used to first process all `extern crate`, then all `use` and then all items can still do so, it
just has to filter out the relevant items from the whole module body, rather then from sequential sections of it.
And any usability reasons for keeping the order can be better addressed with conventions and lints, rather than hard parser rules.

(The exception here are the special cased prelude, and globs and macros, which are feature gated and out of scope for this proposal)

As it is, today the ordering rule is a unnecessary complication, as it routinely causes beginner to stumble over things like this:

```rust
mod foo;
use foo::bar; // ERROR: Imports have to precede items
```

In addition, it doesn't even prevent certain patterns, as it is possible to work around the order restriction by using a submodule:

```rust
struct Foo;
// One of many ways to expose the crate out of order:
mod bar { extern crate bar; pub use self::bar::x; pub use self::bar::y; ... }
```

Which with this RFC implemented would be identical to

```rust
struct Foo;
extern crate bar;
```

Another use case are item macros/attributes that want to automatically include their their crate dependencies.
This is possible by having the macro expand to an item that links to the needed crate, eg like this:

```rust
#[my_attribute]
struct UserType;
```

Expands to:

```rust
struct UserType;
extern crate "MyCrate" as <gensymb>
impl <gensymb>::MyTrait for UserType { ... }
```

With the order restriction still in place, this requires the sub module workaround, which is unnecessary verbose.

As an example, [gfx-rs](https://github.com/gfx-rs/gfx-rs) currently employs this strategy.

## Allow `pub extern crate` as opposed to only private ones.

`extern crate` semantically is somewhere between `use`ing a module, and declaring one with `mod`,
and is identical to both as far as as the module path to it is considered.
As such, its surprising that its not possible to declare a `extern crate` as public,
even though you can still make it so with an reexport:

```rust

mod foo {
    extern crate "bar" as bar_;
    pub use bar_ as bar;
}

```

While its generally not neccessary to export a extern library directly, the need for it does arise
occasionally during refactorings of huge crate collections,
generally if a public module gets turned into its own crate.

As an example,the author recalls stumbling over it during a refactoring of gfx-rs.

## Allow `extern crate` in blocks/functions, and not just in modules.

Similar to the point above, its currently possible to both import and declare a module in a
block expression or function body, but not to link to an library:

```rust
fn foo() {
    let x = {
        extern crate qux; // ERROR: Extern crate not allowed here
        use bar::baz;     // OK
        mod bar { ... };  // OK
        qux::foo()
    };
}
```

This is again a unnecessary restriction considering that you can declare modules and imports there,
and thus can make an extern library reachable at that point:

```rust
fn foo() {
    let x = {
        mod qux { extern crate "qux" as qux_; pub use self::qux_ as qux; }
        qux::foo()
    };
}
```

This again benefits macros and gives the developer the power to place external dependencies
only needed for a single function lexically near it.

## General benefits

In general, the simplification and freedom added by these changes
would positively effect the docs of Rusts module system (which is already often regarded as too complex by outsiders),
and possibly admit other simplifications or RFCs based on the now-equality of view items and items in the module system.

(As an example, the author is considering an RFC about merging the `use` and `type` features;
by lifting the ordering restriction they become more similar and thus more redundant)

This also does not have to be a 1.0 feature, as it is entirely backwards compatible to implement,
and strictly allows more programs to compile than before.
However, as alluded to above it might be a good idea for 1.0 regardless

# Detailed design

- Remove the ordering restriction from resolve
- If necessary, change resolve to look in the whole scope block for view items, not just in a prefix of it.
- Make `pub extern crate` parse and teach privacy about it
- Allow `extern crate` view items in blocks

# Drawbacks

- The source of names in scope might be harder to track down
- Similarly, it might become confusing to see when a library dependency exist.

However, these issues already exist today in one form or another, and can be addressed by proper
docs that make library dependencies clear, and by the fact that definitions are generally greppable in a file.

# Alternatives

As this just cleans up a few aspects of the module system, there isn't really an alternative
apart from not or only partially implementing it.

By not implementing this proposal, the module system remains more complex for the user than necessary.

# Unresolved questions

- Inner attributes occupy the same syntactic space as items and view items, and are currently
  also forced into a given order by needing to be written first.
  This is also potentially confusing or restrictive for the same reasons as for the view items
  mentioned above, especially in regard to the build-in crate attributes, and has one big issue:
  It is currently not possible to load a syntax extension
  that provides an crate-level attribute, as with the current macro system this would have to be written like this:

  ```
  #[phase(plugin)]
  extern crate mycrate;
  #![myattr]
  ```

  Which is impossible to write due to the ordering restriction.
  However, as attributes and the macro system are also not finalized, this has not been included in
  this RFC directly.
- This RFC does also explicitly not talk about wildcard imports and macros in regard to resolution,
  as those are feature gated today and likely subject to change. In any case, it seems unlikely that
  they will conflict with the changes proposed here, as macros would likely follow
  the same module system rules where possible, and wildcard imports would
  either be removed, or allowed in a way that doesn't conflict with explicitly imported names to
  prevent compilation errors on upstream library changes (new public item may not conflict with downstream items).
