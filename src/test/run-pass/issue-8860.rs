// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut DROP: int = 0i;
static mut DROP_S: int = 0i;
static mut DROP_T: int = 0i;

struct S;
impl Drop for S {
    fn drop(&mut self) {
        unsafe {
            DROP_S += 1;
            DROP += 1;
        }
    }
}
fn f(ref _s: S) {}

struct T { i: int }
impl Drop for T {
    fn drop(&mut self) {
        unsafe {
            DROP_T += 1;
            DROP += 1;
        }
    }
}
fn g(ref _t: T) {}

fn do_test() {
    let s = S;
    f(s);
    unsafe {
        assert_eq!(1, DROP);
        assert_eq!(1, DROP_S);
    }
    let t = T { i: 1 };
    g(t);
    unsafe { assert_eq!(1, DROP_T); }
}

fn main() {
    do_test();
    unsafe {
        assert_eq!(2, DROP);
        assert_eq!(1, DROP_S);
        assert_eq!(1, DROP_T);
    }
}
