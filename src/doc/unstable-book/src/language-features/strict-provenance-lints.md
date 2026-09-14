# `strict_provenance_lints`

The tracking issue for this feature is: [#130351]

[#130351]: https://github.com/rust-lang/rust/issues/130351
-----

The `strict_provenance_lints` feature allows to enable the `implicit_provenance_casts` lint.

## Example

```rust
#![feature(strict_provenance_lints)]
#![warn(implicit_provenance_casts)]

fn main() {
    let dangling = 16_usize as *const u8;
    //~^ WARNING: cast from `usize` to `*const u8` implicitly relies on exposed provenance
    let _addr = dangling as usize;
    //~^ WARNING: cast from `*const u8` to `usize` implicitly exposes pointer provenance
}
```
