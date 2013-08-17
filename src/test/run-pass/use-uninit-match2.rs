// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn foo<T>(o: myoption<T>) -> int {
    let mut x: int;
    match o {
        none::<T> => { fail!(); }
        some::<T>(_t) => { x = 5; }
    }
    return x;
}

enum myoption<T> { none, some(T), }

pub fn main() { info!(5); }
