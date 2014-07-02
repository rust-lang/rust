// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo(int, int, int, int);
struct Bar{a: int, b: int, c: int, d: int}

pub fn main() {
    let Foo(..) = Foo(5, 5, 5, 5);
    let Foo(..) = Foo(5, 5, 5, 5);
    let Bar{..} = Bar{a: 5, b: 5, c: 5, d: 5};
    //let (..) = (5, 5, 5, 5);
    //let Foo(a, b, ..) = Foo(5, 5, 5, 5);
    //let Foo(.., d) = Foo(5, 5, 5, 5);
    //let (a, b, ..) = (5, 5, 5, 5);
    //let (.., c, d) = (5, 5, 5, 5);
    let Bar{b: b, ..} = Bar{a: 5, b: 5, c: 5, d: 5};
    match [5i, 5, 5, 5] {
        [..] => { }
    }
    match [5i, 5, 5, 5] {
        [a, ..] => { }
    }
    match [5i, 5, 5, 5] {
        [.., b] => { }
    }
    match [5i, 5, 5, 5] {
        [a, .., b] => { }
    }
    match [5i, 5, 5] {
        [..] => { }
    }
    match [5i, 5, 5] {
        [a, ..] => { }
    }
    match [5i, 5, 5] {
        [.., a] => { }
    }
    match [5i, 5, 5] {
        [a, .., b] => { }
    }
}
