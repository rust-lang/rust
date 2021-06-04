# `force-warns`

The tracking issue for this feature is: [#85512](https://github.com/rust-lang/rust/issues/85512).

------------------------

This feature allows you to cause any lint to produce a warning even if the lint has a different level by default or another level is set somewhere else. For instance, the `force-warns` option can be used to make a lint (e.g., `dead_code`) produce a warning even if that lint is allowed in code with `#![allow(dead_code)]`.

## Example

```rust,ignore (partial-example)
#![allow(dead_code)]

fn dead_function() {}
// This would normally not produce a warning even though the
// function is not used, because dead code is being allowed

fn main() {}
```

We can force a warning to be produced by providing `--force-warns dead_code` to rustc.
