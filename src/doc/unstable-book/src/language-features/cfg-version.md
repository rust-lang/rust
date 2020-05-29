# `cfg_version`

The tracking issue for this feature is: [#64796]

[#64796]: https://github.com/rust-lang/rust/issues/64796

------------------------

The `cfg_version` feature makes it possible to execute different code
depending on the compiler version.

## Examples

```rust
#![feature(cfg_version)]

#[cfg(version("1.42"))]
fn a() {
    // ...
}

#[cfg(not(version("1.42")))]
fn a() {
    // ...
}

fn b() {
    if cfg!(version("1.42")) {
        // ...
    } else {
        // ...
    }
}
```
