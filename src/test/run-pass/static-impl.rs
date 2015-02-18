// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



pub trait plus {
    fn plus(&self) -> int;
}

mod a {
    use plus;
    impl plus for uint { fn plus(&self) -> int { *self as int + 20 } }
}

mod b {
    use plus;
    impl plus for String { fn plus(&self) -> int { 200 } }
}

trait uint_utils {
    fn str(&self) -> String;
    fn multi<F>(&self, f: F) where F: FnMut(uint);
}

impl uint_utils for uint {
    fn str(&self) -> String {
        self.to_string()
    }
    fn multi<F>(&self, mut f: F) where F: FnMut(uint) {
        let mut c = 0_usize;
        while c < *self { f(c); c += 1_usize; }
    }
}

trait vec_utils<T> {
    fn length_(&self, ) -> uint;
    fn iter_<F>(&self, f: F) where F: FnMut(&T);
    fn map_<U, F>(&self, f: F) -> Vec<U> where F: FnMut(&T) -> U;
}

impl<T> vec_utils<T> for Vec<T> {
    fn length_(&self) -> uint { self.len() }
    fn iter_<F>(&self, mut f: F) where F: FnMut(&T) { for x in self { f(x); } }
    fn map_<U, F>(&self, mut f: F) -> Vec<U> where F: FnMut(&T) -> U {
        let mut r = Vec::new();
        for elt in self {
            r.push(f(elt));
        }
        r
    }
}

pub fn main() {
    assert_eq!(10_usize.plus(), 30);
    assert_eq!(("hi".to_string()).plus(), 200);

    assert_eq!((vec!(1)).length_().str(), "1".to_string());
    let vect = vec!(3, 4).map_(|a| *a + 4);
    assert_eq!(vect[0], 7);
    let vect = (vec!(3, 4)).map_::<uint, _>(|a| *a as uint + 4_usize);
    assert_eq!(vect[0], 7_usize);
    let mut x = 0_usize;
    10_usize.multi(|_n| x += 2_usize );
    assert_eq!(x, 20_usize);
}
