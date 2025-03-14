# Re-exports

Let's start by explaining what are re-exports. To do so, we will use an example where we are
writing a library (named `lib`) with some types dispatched in sub-modules:

```rust
pub mod sub_module1 {
    pub struct Foo;
}
pub mod sub_module2 {
    pub struct AnotherFoo;
}
```

Users can import them like this:

```rust,ignore (inline)
use lib::sub_module1::Foo;
use lib::sub_module2::AnotherFoo;
```

But what if you want the types to be available directly at the crate root or if we don't want the
modules to be visible for users? That's where re-exports come in:

```rust,ignore (inline)
// `sub_module1` and `sub_module2` are not visible outside.
mod sub_module1 {
    pub struct Foo;
}
mod sub_module2 {
    pub struct AnotherFoo;
}
// We re-export both types:
pub use crate::sub_module1::Foo;
pub use crate::sub_module2::AnotherFoo;
```

And now users will be able to do:

```rust,ignore (inline)
use lib::{Foo, AnotherFoo};
```

And since both `sub_module1` and `sub_module2` are private, users won't be able to import them.

Now what's interesting is that the generated documentation for this crate will show both `Foo` and
`AnotherFoo` directly at the crate root, meaning they have been inlined. There are a few rules to
know whether or not a re-exported item will be inlined.

## Inlining rules

If a public item comes from a private module, it will be inlined:

```rust,ignore (inline)
mod private_module {
    pub struct Public;
}
pub mod public_mod {
    // `Public` will inlined here since `private_module` is private.
    pub use super::private_module::Public;
}
// `Public` will not be inlined here since `public_mod` is public.
pub use self::public_mod::Public;
```

Likewise, if an item inherits `#[doc(hidden)]` from any of its ancestors, it will be inlined:

```rust,ignore (inline)
#[doc(hidden)]
pub mod public_mod {
    pub struct Public;
}
// `Public` be inlined since its parent (`public_mod`) has `#[doc(hidden)]`.
pub use self::public_mod::Public;
```

If an item has `#[doc(hidden)]`, it won't be inlined (nor visible in the generated documentation):

```rust,ignore (inline)
// This struct won't be visible.
#[doc(hidden)]
pub struct Hidden;

// This re-export won't be visible.
pub use self::Hidden as InlinedHidden;
```

However, if you still want the re-export itself to be visible, you can add the `#[doc(inline)]`
attribute on it:

```rust,ignore (inline)
// This struct won't be visible.
#[doc(hidden)]
pub struct Hidden;

#[doc(inline)]
pub use self::Hidden as InlinedHidden;
```

In this case, you will have `pub use self::Hidden as InlinedHidden;` in the generated documentation
but no link to the `Hidden` item.

So back to `#[doc(hidden)]`: if you have multiple re-exports and some of them have
`#[doc(hidden)]`, then these ones (and only these) won't appear in the documentation:

```rust,ignore (inline)
mod private_mod {
    /// First
    pub struct InPrivate;
}

/// Second
#[doc(hidden)]
pub use self::private_mod::InPrivate as Hidden;
/// Third
pub use self::Hidden as Visible;
```

In this case, `InPrivate` will be inlined as `Visible`. However, its documentation will be
`First Third` and not `First Second Third` because the re-export with `Second` as documentation has
`#[doc(hidden)]`, therefore, all its attributes are ignored.

## Inlining with `#[doc(inline)]`

You can use the `#[doc(inline)]` attribute if you want to force an item to be inlined:

```rust,ignore (inline)
pub mod public_mod {
    pub struct Public;
}
#[doc(inline)]
pub use self::public_mod::Public;
```

With this code, even though `public_mod::Public` is public and present in the documentation, the
`Public` type will be present both at the crate root and in the `public_mod` module.

## Preventing inlining with `#[doc(no_inline)]`

On the opposite of the `#[doc(inline)]` attribute, if you want to prevent an item from being
inlined, you can use `#[doc(no_inline)]`:

```rust,ignore (inline)
mod private_mod {
    pub struct Public;
}
#[doc(no_inline)]
pub use self::private_mod::Public;
```

In the generated documentation, you will see a re-export at the crate root and not the type
directly.

## Attributes

When an item is inlined, its doc comments and most of its attributes will be inlined along with it:

```rust,ignore (inline)
mod private_mod {
    /// First
    #[cfg(a)]
    pub struct InPrivate;
    /// Second
    #[cfg(b)]
    pub use self::InPrivate as Second;
}

/// Third
#[doc(inline)]
#[cfg(c)]
pub use self::private_mod::Second as Visible;
```

In this case, `Visible` will have as documentation `First Second Third` and will also have as `cfg`:
`#[cfg(a, b, c)]`.

[Intra-doc links](./linking-to-items-by-name.md) are resolved relative to where the doc comment is
defined.

There are a few attributes which are not inlined though:
 * `#[doc(alias="")]`
 * `#[doc(inline)]`
 * `#[doc(no_inline)]`
 * `#[doc(hidden)]` (because the re-export itself and its attributes are ignored).

All other attributes are inherited when inlined, so that the documentation matches the behavior if
the inlined item was directly defined at the spot where it's shown.

These rules also apply if the item is inlined with a glob re-export:

```rust,ignore (inline)
mod private_mod {
    /// First
    #[cfg(a)]
    pub struct InPrivate;
}

#[cfg(c)]
pub use self::private_mod::*;
```

Otherwise, the attributes displayed will be from the re-exported item and the attributes on the
re-export itself will be ignored:

```rust,ignore (inline)
mod private_mod {
    /// First
    #[cfg(a)]
    pub struct InPrivate;
}

#[cfg(c)]
pub use self::private_mod::InPrivate;
```

In the above case, `cfg(c)` will not be displayed in the docs.
