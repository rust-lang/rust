// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test

#[test]
#[should_panic = "foo"]
//~^ WARN: attribute must be of the form:
fn test1() {
    panic!();
}

#[test]
#[should_panic(expected)]
//~^ WARN: argument must be of the form:
fn test2() {
    panic!();
}

#[test]
#[should_panic(expect)]
//~^ WARN: argument must be of the form:
fn test3() {
    panic!();
}

#[test]
#[should_panic(expected(foo, bar))]
//~^ WARN: argument must be of the form:
fn test4() {
    panic!();
}

#[test]
#[should_panic(expected = "foo", bar)]
//~^ WARN: argument must be of the form:
fn test5() {
    panic!();
}
