// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(extern_prelude, lang_items, start, alloc)]
#![no_std]

extern crate std as other;

mod foo {
    pub fn test() {
        let x = core::cmp::min(2, 3);
        assert_eq!(x, 2);
    }
}

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    foo::test();
    0
}
