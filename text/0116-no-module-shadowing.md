- Start Date: 2014-06-12
- RFC PR #: https://github.com/rust-lang/rfcs/pull/116
- Rust Issue #: https://github.com/rust-lang/rust/issues/16464

# Summary

Remove or feature gate the shadowing of view items on the same scope level, in order to have less
complicated semantic and be more future proof for module system changes or experiments.

This means the names brought in scope by `extern crate` and `use` may never collide with
each other, nor with any other item (unless they live in different namespaces).
Eg, this will no longer work:

```rust
extern crate foo;
use foo::bar::foo; // ERROR: There is already a module `foo` in scope
```

Shadowing would still be allowed in case of lexical scoping, so this continues to work:

```rust
extern crate foo;

fn bar() {
    use foo::bar::foo; // Shadows the outer foo

    foo::baz();
}

```
# Definitions
Due to a certain lack of official, clearly defined semantics and terminology, a list of relevant
definitions is included:

- __Scope__
  A _scope_ in Rust is basically defined by a block, following the rules of lexical
  scoping:

  ```
  scope 1 (visible: scope 1)
  {
        scope 1-1 (visible: scope 1, scope 1-1)
        {
            scope 1-1-1 (visible: scope 1, scope 1-1, scope 1-1-1)
        }
        scope 1-1
        {
            scope 1-1-2
        }
        scope 1-1
  }
  scope 1
  ```

  Blocks include block expressions, `fn` items and `mod` items, but not things like
  `extern`, `enum` or `struct`. Additionally, `mod` is special in that it isolates itself from
  parent scopes.
- __Scope Level__
  Anything with the same name in the example above is on the same scope level.
  In a scope level, all names defined in parent scopes are visible, but can be shadowed
  by a new definition with the same name, which will be in scope for that scope itself and all its
  child scopes.
- __Namespace__
  Rust has different namespaces, and the scoping rules apply to each one separately.
  The exact number of different namespaces is not well defined, but they are roughly
  - types (`enum Foo {}`)
  - modules (`mod foo {}`)
  - item values (`static FOO: uint = 0;`)
  - local values (`let foo = 0;`)
  - lifetimes (`impl<'a> ...`)
  - macros (`macro_rules! foo {...}`)
- __Definition Item__
  Declarations that create new entities in a crate are called (by the author)
  definition items. They include `struct`, `enum`, `mod`, `fn`, etc.
  Each of them creates a name in the type, module, item value or macro namespace in the same
  scope level they are written in.
- __View Item__
  Declarations that just create aliases to existing declarations in a crate are called
  view items. They include `use` and `extern crate`, and also create a name in the type,
  module, item value or macro namespace in the same scope level they are written in.
- __Item__
  Both definition items and view items together are collectively called items.
- __Shadowing__
  While the principle of shadowing exists in all namespaces, there are different forms of it:
  - item-style: Declarations shadow names from outer scopes, and are visible everywhere in their
    own, including lexically before their own definition.
    This requires there to be only one definition with the same name and namespace per scope level.
    Types, modules, item values and lifetimes fall under these rules.
  - sequentially: Declarations shadow names that are lexically before them, both in parent scopes
    and their own. This means you can reuse the same name in the same scope, but a definition
    will not be visibly before itself. This is how local values and macros work.
    (Due to sequential code execution and parsing, respectively)
  - _view item_:
    A special case exists with view items; In the same scope level,
    `extern crate` creates entries in the module namespace, which are shadowable by names created
    with `use`, which are shadowable with any definition item.
    __The singular goal of this RFC is to remove this shadowing behavior of view items__

# Motivation

As explained above, what is currently visible under which namespace in a given scope is determined
by a somewhat complicated three step process:

1. First, every `extern crate` item creates a name in the module namespace.
2. Then, every `use` can create a name in any namespace,
   shadowing the `extern crate` ones.
3. Lastly, any definition item can shadow any name brought in scope by both `extern crate` and `use`.

These rules have developed mostly in response to the older, more complicated import system, and
the existence of wildcard imports (`use foo::*`).
In the case of wildcard imports, this shadowing behavior prevents local code from breaking if the
source module gets updated to include new names that happen to be defined locally.

However, wildcard imports are now feature gated, and name conflicts in general can be resolved by
using the renaming feature of `extern crate` and `use`, so in the current non-gated state
of the language there is no need for this shadowing behavior.

Gating it off opens the door to remove it altogether in a backwards compatible way, or to
re-enable it in case wildcard imports are officially supported again.

It also makes the mental model around items simpler: Any shadowing of items happens through
lexical scoping only, and every item can be considered unordered and mutually recursive.

If this RFC gets accepted, a possible next step would be a RFC to lift the ordering restriction
between `extern crate`, `use` and definition items, which would make them truly behave the same in
regard to shadowing and the ability to be reordered. It would also lift the weirdness of
`use foo::bar; mod foo;`.

Implementing this RFC would also not change anything about how name resolution works, as its just
a tightening of the existing rules.

# Drawbacks

- Feature gating import shadowing might break some code using `#[feature(globs)]`.
- The behavior of `libstd`s prelude becomes more magical if it still allows shadowing,
  but this could be de-magified again by a new feature, see below in unresolved questions.
- Or the utility of `libstd`s prelude becomes more restricted if it doesn't allow shadowing.

# Detailed design

A new feature gate `import_shadowing` gets created.

During the name resolution phase of compilation, every time the compiler detects a shadowing
between `extern crate`, `use` and definition items in the same scope level,
it bails out unless the feature gate got enabled. This amounts to two rules:

- Items in the same scope level and either the type, module, item value or lifetime namespace
  may not shadow each other in the respective namespace.
- Items may shadow names from outer scopes in any namespace.

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
  as defining things with the same name as std exports is not recommended anyway, and this would
  nicely enforce that. It would however mean that the prelude can not change without breaking
  backwards compatibility, which might be too restricting.

  A compromise would be to specialize wildcard imports into a new `prelude use` feature, which
  has the explicit properties of being shadow-able and using a wildcard import. `libstd`s prelude
  could then simply use that, and users could define and use their own preludes as well.
  But that's a somewhat orthogonal feature, and should be discussed in its own RFC.

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
