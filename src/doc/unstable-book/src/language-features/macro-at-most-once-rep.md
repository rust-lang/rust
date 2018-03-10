# `macro_at_most_once_rep`

The tracking issue for this feature is: TODO(mark-i-m)

With this feature gate enabled, one can use `?` as a Kleene operator meaning "0
or 1 repetitions" in a macro definition. Previously only `+` and `*` were allowed.

For example:

```rust
#![feature(macro_at_most_once_rep)]

macro_rules! foo {
    (something $(,)?) // `?` indicates `,` is "optional"...
        => {}
}
```

------------------------

