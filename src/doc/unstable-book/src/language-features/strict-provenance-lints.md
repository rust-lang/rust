# `strict_provenance_lints`

The tracking issue for this feature is: [#95228]

[#95228]: https://github.com/rust-lang/rust/issues/95228
-----

The `strict_provenance_lints` feature allows to enable the `fuzzy_provenance_casts` and `lossy_provenance_casts` lints.
These lint on casts between integers and pointers, that are recommended against or invalid in the strict provenance model.

## Example

```rust
#![feature(strict_provenance_lints)]
#![warn(fuzzy_provenance_casts)]

fn main() {
    let _dangling = 16_usize as *const u8;
    //~^ WARNING: strict provenance disallows casting integer `usize` to pointer `*const u8`
}
```
