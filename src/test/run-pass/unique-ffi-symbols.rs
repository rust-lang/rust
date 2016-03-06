// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We used to have a __rust_abi shim that resulted in duplicated symbols
// whenever the item path wasn't enough to disambiguate between them.
fn main() {
    let a = {
        extern fn good() -> i32 { return 0; }
        good as extern fn() -> i32
    };
    let b = {
        extern fn good() -> i32 { return 5; }
        good as extern fn() -> i32
    };

    assert!(a != b);
    assert_eq!((a(), b()), (0, 5));
}
