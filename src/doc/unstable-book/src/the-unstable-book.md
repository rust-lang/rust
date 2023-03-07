# The Unstable Book

Welcome to the Unstable Book! This book consists of a number of chapters,
each one organized by a "feature flag." That is, when using an unstable
feature of Rust, you must use a flag, like this:

```rust
#![feature(box_syntax)]

fn main() {
    let five = box 5;
}
```

The `box_syntax` feature [has a chapter][box] describing how to use it.

[box]: language-features/box-syntax.md

Because this documentation relates to unstable features, we make no guarantees
that what is contained here is accurate or up to date. It's developed on a
best-effort basis. Each page will have a link to its tracking issue with the
latest developments; you might want to check those as well.
