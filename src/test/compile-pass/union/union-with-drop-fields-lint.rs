// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]
#![allow(dead_code)]
#![allow(unions_with_drop_fields)]

union U {
    a: u8, // OK
}

union W {
    a: String, // OK
    b: String, // OK
}

struct S(String);

// `S` doesn't implement `Drop` trait, but still has non-trivial destructor
union Y {
    a: S, // OK
}

// We don't know if `T` is trivially-destructable or not until trans
union J<T> {
    a: T, // OK
}

union H<T: Copy> {
    a: T, // OK
}

fn main() {}
