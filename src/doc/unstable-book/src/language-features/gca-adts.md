# gca_adts

Enables ADTs under the generic const args family of features (e.g. `gca!` macro).

The tracking issue for this feature is: [#163420]

[#163420]: https://github.com/rust-lang/rust/issues/163420

## Examples

```rust
#![feature(gca_min_const_items, min_adt_const_params, gca_adts)]

use std::gca;

struct S<const A: [u32; 2]>;

fn main() {
    let _: S<gca!([1, 2])> = S::<gca!([1, 2])>;
}
```
