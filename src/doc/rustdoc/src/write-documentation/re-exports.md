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

```rust,ignore
use lib::sub_module1::Foo;
use lib::sub_module2::AnotherFoo;
```

But what if you want the types to be available directly at the crate root or if we don't want the
modules to be visible for users? That's where re-exports come in:

```rust
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

```rust,ignore
use lib::{Foo, AnotherFoo};
```

And since both `sub_module1` and `sub_module2` are private, users won't be able to import them.

Now what's interesting is that the generated documentation for this crate will show both `Foo` and
`AnotherFoo` directly at the crate root, meaning they have been inlined. There are a few rules to
know whether or not a re-exported item will be inlined.

## Inlining rules

If a public item comes from a private module, it will be inlined:

```rust
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

Likewise, if an item inherits has `#[doc(hidden)]` or inherits it (from any of its parents), it
will be inlined:

```rust
#[doc(hidden)]
pub mod public_mod {
    pub struct Public;
}

#[doc(hidden)]
pub struct Hidden;

// `Public` be inlined since its parent (`public_mod`) has `#[doc(hidden)]`.
pub use self::public_mod::Public;
// `Hidden` be inlined since it has `#[doc(hidden)]`.
pub use self::Hidden;
```

## Inlining with `#[doc(inline)]`

You can use the `#[doc(inline)]` attribute if you want to force an item to be inlined:

```rust
pub mod public_mod {
    pub struct Public;
}

#[doc(inline)]
pub use self::public_mod::Public;
```

With this code, even though `public_mod::Public` is public and present in the documentation, the
`Public` type will be present both at the crate root and in the `public_mod` module.
