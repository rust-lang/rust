# `c_size_t`

The tracking issue for this feature is: [#88345]

[#88345]: https://github.com/rust-lang/rust/issues/88345
-----

The `c_size_t` feature allows to enable C FFI types `size_t` `ssize_t` `intptr_t` `uintptr_t` `ptrdiff_t`.

## Example

```rust
#![feature(c_size_t)]

use std::ffi::{TaggedPointer, c_uintptr_t};

fn main() {
    let ptr = core::ptr::with_exposed_provenance(16_usize);
    let _ptr_uintptr_t = c_uintptr_t::new(ptr);
}
```
