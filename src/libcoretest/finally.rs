// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::finally::{try_finally, Finally};
use std::task::failing;

#[test]
fn test_success() {
    let mut i = 0i;
    try_finally(
        &mut i, (),
        |i, ()| {
            *i = 10;
        },
        |i| {
            assert!(!failing());
            assert_eq!(*i, 10);
            *i = 20;
        });
    assert_eq!(i, 20);
}

#[test]
#[should_fail]
fn test_fail() {
    let mut i = 0i;
    try_finally(
        &mut i, (),
        |i, ()| {
            *i = 10;
            fail!();
        },
        |i| {
            assert!(failing());
            assert_eq!(*i, 10);
        })
}

#[test]
fn test_retval() {
    let mut closure: || -> int = || 10;
    let i = closure.finally(|| { });
    assert_eq!(i, 10);
}

#[test]
fn test_compact() {
    fn do_some_fallible_work() {}
    fn but_always_run_this_function() { }
    let mut f = do_some_fallible_work;
    f.finally(but_always_run_this_function);
}
