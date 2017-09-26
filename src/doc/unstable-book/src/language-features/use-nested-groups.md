# `use_nested_groups`

The tracking issue for this feature is: [#44494]

[#44494]: https://github.com/rust-lang/rust/issues/44494

------------------------

The `use_nested_groups` feature allows you to import multiple items from a
complex module tree easily, by nesting different imports in the same
declaration. For example:

```rust
#![feature(use_nested_groups)]
# #![allow(unused_imports, dead_code)]
#
# mod foo {
#     pub mod bar {
#         pub type Foo = ();
#     }
#     pub mod baz {
#         pub mod quux {
#             pub type Bar = ();
#         }
#     }
# }

use foo::{
    bar::{self, Foo},
    baz::{*, quux::Bar},
};
#
# fn main() {}
```

## Snippet for the book's new features appendix

When stabilizing, add this to
`src/doc/book/second-edition/src/appendix-07-newest-features.md`:

### Nested groups in `use` declarations

If you have a complex module tree with many different submodules and you need
to import a few items from each one, it might be useful to group all the
imports in the same declaration to keep your code clean and avoid repeating the
base modules' name.

The `use` declaration supports nesting to help you in those cases, both with
simple imports and glob ones. For example this snippets imports `bar`, `Foo`,
all the items in `baz` and `Bar`:

```rust
# #![feature(use_nested_groups)]
# #![allow(unused_imports, dead_code)]
#
# mod foo {
#     pub mod bar {
#         pub type Foo = ();
#     }
#     pub mod baz {
#         pub mod quux {
#             pub type Bar = ();
#         }
#     }
# }
#
use foo::{
    bar::{self, Foo},
    baz::{*, quux::Bar},
};
#
# fn main() {}
```

## Updated reference

When stabilizing, replace the shortcut list in
`src/doc/reference/src/items/use-declarations.md` with this updated one:

* Simultaneously binding a list of paths with a common prefix, using the
  glob-like brace syntax `use a::b::{c, d, e::f, g::h::i};`
* Simultaneously binding a list of paths with a common prefix and their common
  parent module, using the `self` keyword, such as `use a::b::{self, c, d::e};`
* Rebinding the target name as a new local name, using the syntax `use p::q::r
  as x;`. This can also be used with the last two features:
  `use a::b::{self as ab, c as abc}`.
* Binding all paths matching a given prefix, using the asterisk wildcard syntax
  `use a::b::*;`.
* Nesting groups of the previous features multiple times, such as
  `use a::b::{self as ab, c d::{*, e::f}};`
