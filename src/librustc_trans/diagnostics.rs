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

```
extern "rust-intrinsic" {
    fn return_address() -> *const u8;
}

pub unsafe fn by_value() -> i32 {
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
extern "rust-intrinsic" {
    fn return_address() -> *const u8;
}

pub unsafe fn by_pointer() -> String {
    let _ = return_address();
    String::new() // ok!
}
```
"##,

E0511: r##"
Invalid monomorphization of an intrinsic function was used. Erroneous code
example:

```
extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

unsafe { simd_add(0, 1); }
// error: invalid monomorphization of `simd_add` intrinsic
```

The generic type has to be a SIMD type. Example:

```
#[repr(simd)]
#[derive(Copy, Clone)]
struct i32x1(i32);

extern "platform-intrinsic" {
    fn simd_add<T>(a: T, b: T) -> T;
}

unsafe { simd_add(i32x1(0), i32x1(1)); } // ok!
```
"##,

E0512: r##"
Transmute with two differently sized types was attempted. Erroneous code
example:

```
extern "rust-intrinsic" {
    pub fn ctpop8(x: u8) -> u8;
}

fn main() {
    unsafe { ctpop8(::std::mem::transmute(0u16)); }
    // error: transmute called with differently sized types
}
```

Please use types with same size or use the expected type directly. Example:

```
extern "rust-intrinsic" {
    pub fn ctpop8(x: u8) -> u8;
}

fn main() {
    unsafe { ctpop8(::std::mem::transmute(0i8)); } // ok!
    // or:
    unsafe { ctpop8(0u8); } // ok!
}
```
"##,

E0515: r##"
A constant index expression was out of bounds. Erroneous code example:

```
let x = &[0, 1, 2][7]; // error: const index-expr is out of bounds
```

Please specify a valid index (not inferior to 0 or superior to array length).
Example:

```
let x = &[0, 1, 2][2]; // ok
```
"##,

}
