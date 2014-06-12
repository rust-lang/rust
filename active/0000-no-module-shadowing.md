- Start Date: 2014-06-12
- RFC PR #:
- Rust Issue #:

# Summary

Feature gate the shadowing of all namespaces between `extern crate`, `use` and items in the
same scope, in order to restrict the module system to something more open to experimentation
and changes post 1.0.

# Motivation

Currently, what is visible under which namespace in a given module is determined by a
somewhat complicated three step process:

1. First, every `extern crate` item creates a name in the module namespace.
2. Then, every `use` can create a name in any of the three namespaces,
   where the module ones shadow the `extern crate` ones.
3. Lastly, any declaration can shadow any name brought in scope by both `extern crate` and `use`.

These rules have developed mostly in response to the older, more complicated import system, and
the existence of wildcard imports (`use foo::*`), which can cause the problem that user code breaks
if a used crate gets updated to include a definition name the user has used himself.

However, wildcard imports are now feature gated, and name conflicts can be resolved by using the
renaming feature of `extern crate` and `use`, so in the current state of the language there is no
need for this shadowing behavior.

Gating it off opens the door to remove it altogether in a backwards compatible way, or to
re-enable it in case globs get enabled again.

# Drawbacks

- Feature gating import shadowing might break some code using `#[feature(globs)]`.
- The behavior of `libstd`s prelude either becomes more magical if it still allows shadowing,
  or more restricted if it doesn't allow shadowing.

# Detailed design

A new feature gate `import_shadowing` gets created.

During the name resolution phase of compilation, every time the compiler detects a shadowing
between `extern crate`, `use` and declarations in the same scope,
it bails out unless the feature gate got enabled.

Just like for the `globs` feature, the `libstd` prelude import would be preempt from this,
and still be allowed to be shadowed.

# Alternatives

The alternative is to do nothing, and risk running into a backwards compatibility hazard,
or committing to make a final design decision around the whole module system before 1.0 gets
released.

# Unresolved questions

- It is unclear how the `libstd` preludes fits into this.

  On the one hand, it basically acts like a hidden `use std::prelude::*;` import
  which ignores the `globs` feature, so it could simply also ignore the
  `import_shadowing` feature as well, and the rule becomes that the prelude is a magic
  compiler feature that injects imports into every module but doesn't prevent the user
  from taking the same names.

  On the other hand, it is also thinkable to simply forbid shadowing of prelude items as well,
  as defining things with the same name as std exports is unrecommended anyway, and this would
  nicely enforce that. It would however mean that the prelude can not change without breaking
  backwards compatibility, which might be too restricting.

  A compromise would be to specialize wildcard imports into a new `prelude use` feature, which
  has the explicit properties of being shadow-able and using a wildcard import. `libstd`s prelude
  could then simply use that, and users could define and use their own preludes as well.
  But that's a somewhat orthogonal feature, and should be discussed in its own RFC.

- Should scoped declarations fall under these rules as well?

  Currently, you can also shadow declarations and imports by using lexical scopes. For example,
  each struct definition shadows the prior one here:
  ```rust
struct Foo(());
fn main() {
        struct Foo(());
        static FOO: () = {
            struct Foo(());
            ()
        };
}
  ```
  That feature will probably stay, but there might be consistency problems
  or interactions with this proposal, which is why it is included in the discussion here.

- Interaction with overlapping imports.

  Right now its legal to write this:
  ```rust
fn main() {
        use Bar = std::result::Result;
        use Bar = std::option::Option;
        let x: Bar<uint> = None;
}
  ```
  where the latter `use` shadows the former. This would have to be forbidden as well,
  however the current semantic seems like a accident anyway.
