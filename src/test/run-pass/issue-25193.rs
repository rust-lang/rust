// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(scoped_tls)]

trait TestTrait {
    fn test(&self) -> bool;
}

struct TestStruct;

impl TestTrait for TestStruct {
    fn test(&self) -> bool {
        true
    }
}

scoped_thread_local!(static TEST_SLICE: [u32]);
scoped_thread_local!(static TEST_TRAIT: TestTrait);

pub fn main() {
    TEST_SLICE.set(&[0; 10], || {
        TEST_SLICE.with(|slice| {
            assert_eq!(slice, [0; 10]);
        });
    });
    TEST_TRAIT.set(&TestStruct, || {
        TEST_TRAIT.with(|traitObj| {
            assert!(traitObj.test());
        });
    });
}
