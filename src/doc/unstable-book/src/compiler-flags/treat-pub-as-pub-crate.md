# `treat-pub-as-pub-crate`

The `-Ztreat-pub-as-pub-crate` flag causes the compiler to treat all `pub` items in the crate as if they were `pub(crate)`. This is useful for finding dead code in binary crates where public items are not intended to be exported to other crates.

When this flag is enabled, the dead code lint will warn about public items that are not used within the crate (unless they are the entry point, like `main`).

## Example

Consider the following code in a binary crate:

```rust
pub fn unused_pub_fn() {}

fn main() {
    println!("Hello, world!");
}
```

By default, `rustc` assumes that `unused_pub_fn` might be used by other crates linking to this binary, so it does not warn about it being unused.

With `-Ztreat-pub-as-pub-crate`, `rustc` will emit a warning:

```text
warning: function `unused_pub_fn` is never used
 --> src/main.rs:1:8
  |
1 | pub fn unused_pub_fn() {}
  |        ^^^^^^^^^^^^^
  |
  = note: `#[warn(dead_code)]` on by default
```
