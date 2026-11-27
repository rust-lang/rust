# `cfg_target_object_format`

The tracking issue for this feature is: [#152586]

[#152586]: https://github.com/rust-lang/rust/issues/152586

------------------------

The `cfg_target_object_format` feature makes it possible to execute different code
depending on the current target's object file format.

## Examples

```rust
#![feature(cfg_target_object_format)]

#[cfg(target_object_format = "elf")]
fn a() {
    // ...
}

#[cfg(target_object_format = "mach-o")]
fn a() {
    // ...
}

fn b() {
    if cfg!(target_object_format = "wasm") {
        // ...
    } else {
        // ...
    }
}
```
