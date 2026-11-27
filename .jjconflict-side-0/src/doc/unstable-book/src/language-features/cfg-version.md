# `cfg_version`

The tracking issue for this feature is: [#64796]

[#64796]: https://github.com/rust-lang/rust/issues/64796

------------------------

The `cfg_version` feature makes it possible to execute different code
depending on the compiler version. It will return true if the compiler
version is greater than or equal to the specified version.

## Examples

```rust
#![feature(cfg_version)]

#[cfg(version("1.42"))] // 1.42 and above
fn a() {
    // ...
}

#[cfg(not(version("1.42")))] // 1.41 and below
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
