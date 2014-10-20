// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15877

pub trait Clone2 {
    /// Returns a copy of the value. The contents of owned pointers
    /// are copied to maintain uniqueness, while the contents of
    /// managed pointers are not copied.
    fn clone(&self) -> Self;
}

trait Getter<T: Clone> {
    fn do_get(&self) -> T;

    fn do_get2(&self) -> (T, T) {
        let x = self.do_get();
        (x.clone(), x.clone())
    }

}

impl Getter<int> for int {
    fn do_get(&self) -> int { *self }
}

impl<T: Clone> Getter<T> for Option<T> {
    fn do_get(&self) -> T { self.as_ref().unwrap().clone() }
}


pub fn main() {
    assert_eq!(3.do_get2(), (3, 3));
    assert_eq!(Some("hi".to_string()).do_get2(), ("hi".to_string(), "hi".to_string()));
}
