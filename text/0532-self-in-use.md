- Start Date: 2014-12-19
- RFC PR: [532](https://github.com/rust-lang/rfcs/pull/532)
- Rust Issue: [20361](https://github.com/rust-lang/rust/issues/20361)

# Summary

This RFC proposes the `mod` keyword used to refer
the immediate parent namespace in `use` items (`use a::b::{mod, c}`)
to be changed to `self`.

# Motivation

While this looks fine:

````rust
use a::b::{mod, c};

pub mod a {
    pub mod b {
        pub type c = ();
    }
}
````

This looks strange, since we are not really importing a module:

````rust
use Foo::{mod, Bar, Baz};

enum Foo { Bar, Baz }
````

RFC #168 was written when there was no namespaced `enum`,
therefore the choice of the keyword was suboptimal.

# Detailed design

This RFC simply proposes to use `self` in place of `mod`.
This should amount to one line change to the parser,
possibly with a renaming of relevant AST node (`PathListMod`).

# Drawbacks

`self` is already used to denote a relative path in the `use` item.
While they can be clearly distinguished
(any use of `self` proposed in this RFC will appear inside braces),
this can cause some confusion to beginners.

# Alternatives

Don't do this.
Simply accept that `mod` also acts as a general term for namespaces.

Allow `enum` to be used in place of `mod` when the parent item is `enum`.
This clearly expresses the intent and it doesn't reuse `self`.
However, this is not very future-proof for several reasons.

* Any item acting as a namespace would need a corresponding keyword.
  This is backward compatible but cumbersome.
* If such namespace is not defined with an item but only implicitly,
  we may not have a suitable keyword to use.
* We currently import all items sharing the same name (e.g. `struct P(Q);`),
  with no way of selectively importing one of them by the item type.
  An explicit item type in `use` will imply that we *can* selectively import,
  while we actually can't.

# Unresolved questions

None.
