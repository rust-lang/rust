// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {

E0510: r##"
`return_address` was used in an invalid context. Erroneous code example:

```ignore
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn return_address() -> *const u8;
}

unsafe fn by_value() -> i32 {
    let _ = return_address();
    // error: invalid use of `return_address` intrinsic: function does
    //        not use out pointer
    0
}
```

Return values may be stored in a return register(s) or written into a so-called
out pointer. In case the returned value is too big (this is
target-ABI-dependent and generally not portable or future proof) to fit into
the return register(s), the compiler will return the value by writing it into
space allocated in the caller's stack frame. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn return_address() -> *const u8;
}

unsafe fn by_pointer() -> String {
    let _ = return_address();
    String::new() // ok!
}
```
"##,

E0511: r##"
Invalid monomorphization of an intrinsic function was used. Erroneous code
example:

```ignore
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

unsafe { simd_add(0, 1); }
// error: invalid monomorphization of `simd_add` intrinsic
```

The generic type has to be a SIMD type. Example:

```
#![feature(repr_simd)]
#![feature(platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x1(i32);

extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

unsafe { simd_add(i32x1(0), i32x1(1)); } // ok!
```
"##,
}
