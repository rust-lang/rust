# `macro_at_most_once_rep`

NOTE: This feature is only available in the 2018 Edition.

The tracking issue for this feature is: #48075

With this feature gate enabled, one can use `?` as a Kleene operator meaning "0
or 1 repetitions" in a macro definition. Previously only `+` and `*` were allowed.

For example:

```rust,ignore
#![feature(macro_at_most_once_rep)]

macro_rules! foo {
    (something $(,)?) // `?` indicates `,` is "optional"...
        => {}
}
```

------------------------

