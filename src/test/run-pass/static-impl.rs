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

pub trait plus {
    fn plus() -> int;
}

mod a {
    use plus;
    impl plus for uint { fn plus() -> int { self as int + 20 } }
}

mod b {
    use plus;
    impl plus for ~str { fn plus() -> int { 200 } }
}

trait uint_utils {
    fn str() -> ~str;
    fn multi(f: fn(uint));
}

impl uint_utils for uint {
    fn str() -> ~str { uint::to_str(self) }
    fn multi(f: fn(uint)) {
        let mut c = 0u;
        while c < self { f(c); c += 1u; }
    }
}

trait vec_utils<T> {
    fn length_() -> uint;
    fn iter_(f: fn(&T));
    fn map_<U:Copy>(f: fn(&T) -> U) -> ~[U];
}

impl<T> vec_utils<T> for ~[T] {
    fn length_() -> uint { vec::len(self) }
    fn iter_(f: fn(&T)) { for self.each |x| { f(x); } }
    fn map_<U:Copy>(f: fn(&T) -> U) -> ~[U] {
        let mut r = ~[];
        for self.each |elt| { r += ~[f(elt)]; }
        r
    }
}

pub fn main() {
    fail_unless!(10u.plus() == 30);
    fail_unless!((~"hi").plus() == 200);

    fail_unless!((~[1]).length_().str() == ~"1");
    fail_unless!((~[3, 4]).map_(|a| *a + 4 )[0] == 7);
    fail_unless!((~[3, 4]).map_::<uint>(|a| *a as uint + 4u )[0] == 7u);
    let mut x = 0u;
    10u.multi(|_n| x += 2u );
    fail_unless!(x == 20u);
}
