// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::uint;

pub trait plus {
    fn plus(&self) -> int;
}

mod a {
    use plus;
    impl plus for uint { fn plus(&self) -> int { *self as int + 20 } }
}

mod b {
    use plus;
    impl plus for ~str { fn plus(&self) -> int { 200 } }
}

trait uint_utils {
    fn str(&self) -> ~str;
    fn multi(&self, f: &fn(uint));
}

impl uint_utils for uint {
    fn str(&self) -> ~str { self.to_str() }
    fn multi(&self, f: &fn(uint)) {
        let mut c = 0u;
        while c < *self { f(c); c += 1u; }
    }
}

trait vec_utils<T> {
    fn length_(&self, ) -> uint;
    fn iter_(&self, f: &fn(&T));
    fn map_<U>(&self, f: &fn(&T) -> U) -> ~[U];
}

impl<T> vec_utils<T> for ~[T] {
    fn length_(&self) -> uint { self.len() }
    fn iter_(&self, f: &fn(&T)) { for x in self.iter() { f(x); } }
    fn map_<U>(&self, f: &fn(&T) -> U) -> ~[U] {
        let mut r = ~[];
        for elt in self.iter() {
            r.push(f(elt));
        }
        r
    }
}

pub fn main() {
    assert_eq!(10u.plus(), 30);
    assert_eq!((~"hi").plus(), 200);

    assert_eq!((~[1]).length_().str(), ~"1");
    assert_eq!((~[3, 4]).map_(|a| *a + 4 )[0], 7);
    assert_eq!((~[3, 4]).map_::<uint>(|a| *a as uint + 4u )[0], 7u);
    let mut x = 0u;
    10u.multi(|_n| x += 2u );
    assert_eq!(x, 20u);
}
