# `ptrauth_c_discriminator`

The tracking issue for this feature is: [#148640]

[#148640]: https://github.com/rust-lang/rust/issues/148640

------------------------

The `ptrauth_c_discriminator` feature allows setting the pointer authentication discriminator of a
code address held in a static, on targets with function pointer type discrimination (see [the
`aarch64-unknown-linux-pauthtest` target][pauthtest]).

[pauthtest]: ../../rustc/platform-support/aarch64-unknown-linux-pauthtest.html

Rust signs a function pointer with the discriminator of the function it points to. C signs a code
address with the discriminator of the type of the slot holding it, for example 0 for a `void *`.
For a slot whose type differs from the function's, `#[ptrauth_c_discriminator(N)]` makes a
static's initializer sign the code address with `N` instead:

- on a field, it applies to that field in every static whose type contains it;
- on a static, it applies to the static itself, which must be a pointer, a pointer-sized integer,
  or an array of them.

Only a static's initializer is affected. A value written at runtime, or read back as a Rust
function pointer, still needs `core::intrinsics::ptrauth_c_cast`.

Without function pointer type discrimination, the attribute has no effect.

## Examples

```rust
#![feature(ptrauth_c_discriminator)]

use std::ffi::c_void;

extern "C" fn callback(x: i32) -> i32 {
    x
}

// A table passed to C, which reads `handler` as a `void *`.
#[repr(C)]
pub struct Entry {
    pub id: u32,
    #[ptrauth_c_discriminator(0)]
    pub handler: *const c_void,
}

pub static mut ENTRIES: [Entry; 1] = [Entry { id: 1, handler: callback as *const c_void }];

// A static that is itself a `void *`.
#[ptrauth_c_discriminator(0)]
pub static mut CALLBACK: *const c_void = callback as *const c_void;

fn main() {}
```
