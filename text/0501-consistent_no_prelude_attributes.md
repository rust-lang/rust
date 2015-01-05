- Start Date: (2014-12-06)
- RFC PR: https://github.com/rust-lang/rfcs/pull/501
- Rust Issue: https://github.com/rust-lang/rust/issues/20561

# Summary

Make name and behavior of the `#![no_std]` and `#![no_implicit_prelude]` attributes
consistent by renaming the latter to `#![no_prelude]` and having it only apply to the current
module.

# Motivation

Currently, Rust automatically inserts an implicit `extern crate std;` in the crate root that can be
disabled with the `#[no_std]` attribute.

It also automatically inserts an implicit `use std::prelude::*;` in every module that can be
disabled with the `#[no_implicit_prelude]` attribute.

Lastly, if `#[no_std]` is used, all module automatically don't import the prelude, so the
`#[no_implicit_prelude]` attribute is unneeded in those cases.

However, the later attribute is inconsistent with the former in two regards:

- Naming wise, it redundantly contains the word "implicit"
- Semantic wise, it applies to the current module __and all submodules__.

That last one is surprising because normally, whether or not a module contains a certain import
does not affect whether or not a sub module contains a certain import, so you'd expect a attribute
that disables an implicit import to only apply to that module as well.

This behavior also gets in the way in some of the already rare cases where you want to disable the
prelude while still linking to std.

As an example, the author had been made aware of this behavior of `#[no_implicit_prelude]` while
attempting to prototype a variation of the `Iterator` traits, leading to code that looks like this:

```rust
mod my_iter {
    #![no_implicit_prelude]

    trait Iterator<T> { /* ... */ }

    mod adapters {
        /* Tries to access the existing prelude, and fails to resolve */
    }
}
```

While such use cases might be resolved by just requiring an explicit `use std::prelude::*;`
in the submodules, it seems like just making the attribute behave as expected is the better outcome.

Of course, for the cases where you want the prelude disabled for a whole sub tree of modules, it
would now become necessary to add a `#[no_prelude]` attribute in each of them - but that
is consistent with imports in general.

# Detailed design

`libsyntax` needs to be changed to accept both the name `no_implicit_prelude` and `no_prelude` for
the attribute. Then the attributes effect on the AST needs to be changed to not deeply remove all
imports, and all fallout of this change needs to be fixed in order for the new semantic to
bootstrap.

Then a snapshot needs to be made, and all uses of `#[no_implicit_prelude]` can be
changed to `#[no_prelude]` in both the main code base, and user code.

Finally, the old attribute name should emit a deprecated warning, and be removed in time.

# Drawbacks

- The attribute is a rare use case to begin with, so any effort put into this would
  distract from more important stabilization work.

# Alternatives

 - Keep the current behavior
 - Remove the `#[no_implicit_prelude]` attribute all together, instead forcing users to use
   `#[no_std]` in combination with `extern crate std;` and `use std::prelude::*`.
 - Generalize preludes more to allow custom ones, which might superseed the attributes from this RFC.
