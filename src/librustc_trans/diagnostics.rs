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

Returned values are stored in registers. In the case where the returned
type doesn't fit in a register, the function returns `()` and has an
additional input argument, this is a pointer where the result should
be written. Example:

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

E0512: r##"
A transmute was called on types with different sizes. Erroneous code example:

```
extern "rust-intrinsic" {
    pub fn ctpop8(x: u8) -> u8;
}

fn main() {
    unsafe { ctpop8(::std::mem::transmute(0u16)); }
    // error: transmute called on types with different sizes
}
```

Please use types with same size or use the awaited type directly. Example:

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

register_diagnostics! {
    E0511, // invalid monomorphization of `{}` intrinsic
}
