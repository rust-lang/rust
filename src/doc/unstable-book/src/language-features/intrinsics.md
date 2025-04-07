# `intrinsics`

The tracking issue for this feature is: None.

Intrinsics are rarely intended to be stable directly, but are usually
exported in some sort of stable manner. Prefer using the stable interfaces to
the intrinsic directly when you can.

------------------------


## Intrinsics with fallback logic

Many intrinsics can be written in pure rust, albeit inefficiently or without supporting
some features that only exist on some backends. Backends can simply not implement those
intrinsics without causing any code miscompilations or failures to compile.
All intrinsic fallback bodies are automatically made cross-crate inlineable (like `#[inline]`)
by the codegen backend, but not the MIR inliner.

```rust
#![feature(intrinsics)]
#![allow(internal_features)]

#[rustc_intrinsic]
const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}
```

Since these are just regular functions, it is perfectly ok to create the intrinsic twice:

```rust
#![feature(intrinsics)]
#![allow(internal_features)]

#[rustc_intrinsic]
const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}

mod foo {
    #[rustc_intrinsic]
    const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {
        panic!("noisy const dealloc")
    }
}

```

The behaviour on backends that override the intrinsic is exactly the same. On other
backends, the intrinsic behaviour depends on which implementation is called, just like
with any regular function.

## Intrinsics lowered to MIR instructions

Various intrinsics have native MIR operations that they correspond to. Instead of requiring
backends to implement both the intrinsic and the MIR operation, the `lower_intrinsics` pass
will convert the calls to the MIR operation. Backends do not need to know about these intrinsics
at all. These intrinsics only make sense without a body, and can be declared as a `#[rustc_intrinsic]`.
The body is never used, as calls to the intrinsic do not exist anymore after MIR analyses.

## Intrinsics without fallback logic

These must be implemented by all backends.

### `#[rustc_intrinsic]` declarations

These are written without a body:
```rust
#![feature(intrinsics)]
#![allow(internal_features)]

#[rustc_intrinsic]
pub fn abort() -> !;
```
