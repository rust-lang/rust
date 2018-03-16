// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Baz<T> {}

fn foo<T>(x: T) {
    fn bfnr<U, V: Baz<U>, W: Fn()>(y: T) { //~ ERROR E0401
    }
    fn baz<U,
           V: Baz<U>,
           W: Fn()>
           (y: T) { //~ ERROR E0401
    }
    bfnr(x);
}


struct A<T> {
    inner: T,
}

impl<T> Iterator for A<T> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        fn helper(sel: &Self) -> u8 { //~ ERROR E0401
            unimplemented!();
        }
        Some(helper(self))
    }
}

fn main() {
}
