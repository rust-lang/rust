# `lang_items`

The tracking issue for this feature is: None.

------------------------

The `rustc` compiler has certain pluggable operations, that is,
functionality that isn't hard-coded into the language, but is
implemented in libraries, with a special marker to tell the compiler
it exists. The marker is the attribute `#[lang = "..."]` and there are
various different values of `...`, i.e. various different 'lang
items'. Most of them can only be defined once.

Lang items are loaded lazily by the compiler; e.g. if one never uses `Box`
then there is no need to define a function for `exchange_malloc`.
`rustc` will emit an error when an item is needed but not found in the current
crate or any that it depends on.

Some features provided by lang items:

- overloadable operators via traits: the traits corresponding to the
  `==`, `<`, dereferencing (`*`) and `+` (etc.) operators are all
  marked with lang items; those specific four are `eq`, `partial_ord`,
  `deref`/`deref_mut`, and `add` respectively.
- panicking: the `panic` and `panic_impl` lang items, among others.
- stack unwinding: the lang item `eh_personality` is a function used by the
  failure mechanisms of the compiler. This is often mapped to GCC's personality
  function (see the [`std` implementation][personality] for more information),
  but programs which don't trigger a panic can be assured that this function is
  never called. Additionally, a `eh_catch_typeinfo` static is needed for certain
  targets which implement Rust panics on top of C++ exceptions.
- the traits in `core::marker` used to indicate types of
  various kinds; e.g. lang items `sized`, `sync` and `copy`.
- memory allocation, see below.

Most lang items are defined by `core`, but if you're trying to build
an executable without the `std` crate, you might run into the need
for lang item definitions.

[personality]: https://github.com/rust-lang/rust/blob/master/library/std/src/sys/personality/gcc.rs

## Example: Implementing a `Box`

`Box` pointers require two lang items: one for the type itself and one for
allocation. A freestanding program that uses the `Box` sugar for dynamic
allocations via `malloc` and `free`:

```rust,ignore (libc-is-finicky)
#![feature(lang_items, core_intrinsics, rustc_private, panic_unwind, rustc_attrs)]
#![allow(internal_features)]
#![no_std]
#![no_main]

extern crate libc;
extern crate unwind;

use core::ffi::{c_int, c_void};
use core::intrinsics;
use core::panic::PanicInfo;
use core::ptr::NonNull;

pub struct Global; // the global allocator
struct Unique<T>(NonNull<T>);

#[lang = "owned_box"]
pub struct Box<T, A = Global>(Unique<T>, A);

impl<T> Box<T> {
    pub fn new(x: T) -> Self {
        #[rustc_box]
        Box::new(x)
    }
}

impl<T, A> Drop for Box<T, A> {
    fn drop(&mut self) {
        unsafe {
            libc::free(self.0.0.as_ptr() as *mut c_void);
        }
    }
}

#[lang = "exchange_malloc"]
unsafe fn allocate(size: usize, _align: usize) -> *mut u8 {
    let p = libc::malloc(size) as *mut u8;

    // Check if `malloc` failed:
    if p.is_null() {
        intrinsics::abort();
    }

    p
}

#[no_mangle]
extern "C" fn main(_argc: c_int, _argv: *const *const u8) -> c_int {
    let _x = Box::new(1);

    0
}

#[lang = "eh_personality"]
fn rust_eh_personality() {}

#[panic_handler]
fn panic_handler(_info: &PanicInfo) -> ! { intrinsics::abort() }
```

Note the use of `abort`: the `exchange_malloc` lang item is assumed to
return a valid pointer, and so needs to do the check internally.

## List of all language items

An up-to-date list of all language items can be found [here] in the compiler code.

[here]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_hir/src/lang_items.rs
