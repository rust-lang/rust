// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn test_int() {
    fn f() -> isize { 10 }
    assert_eq!(f(), 10);
}

fn test_vec() {
    fn f() -> Vec<isize> { vec![10, 11] }
    let vect = f();
    assert_eq!(vect[1], 11);
}

fn test_generic() {
    fn f<T>(t: T) -> T { t }
    assert_eq!(f(10), 10);
}

fn test_alt() {
    fn f() -> isize { match true { false => { 10 } true => { 20 } } }
    assert_eq!(f(), 20);
}

fn test_if() {
    fn f() -> isize { if true { 10 } else { 20 } }
    assert_eq!(f(), 10);
}

fn test_block() {
    fn f() -> isize { { 10 } }
    assert_eq!(f(), 10);
}

fn test_ret() {
    fn f() -> isize {
        return 10 // no semi

    }
    assert_eq!(f(), 10);
}


// From issue #372
fn test_372() {
    fn f() -> isize { let x = { 3 }; x }
    assert_eq!(f(), 3);
}

fn test_nil() { () }

pub fn main() {
    test_int();
    test_vec();
    test_generic();
    test_alt();
    test_if();
    test_block();
    test_ret();
    test_372();
    test_nil();
}
