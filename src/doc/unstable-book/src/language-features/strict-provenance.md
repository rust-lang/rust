# `strict_provenance`

The tracking issue for this feature is: [#95228]

[#95228]: https://github.com/rust-lang/rust/issues/95228
-----

The `strict_provenance` feature allows to enable the `fuzzy_provenance_casts` and `lossy_provenance_casts` lints.
These lint on casts between integers and pointers, that are recommended against or invalid in the strict provenance model.
The same feature gate is also used for the experimental strict provenance API in `std` (actually `core`).

## Example

```rust
#![feature(strict_provenance)]
#![warn(fuzzy_provenance_casts)]

fn main() {
    let _dangling = 16_usize as *const u8;
    //~^ WARNING: strict provenance disallows casting integer `usize` to pointer `*const u8`
}
```
