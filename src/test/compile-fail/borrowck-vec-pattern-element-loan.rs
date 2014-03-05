// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn a() -> &[int] {
    let vec = vec!(1, 2, 3, 4);
    let vec: &[int] = vec.as_slice(); //~ ERROR does not live long enough
    let tail = match vec {
        [_, ..tail] => tail,
        _ => fail!("a")
    };
    tail
}

fn b() -> &[int] {
    let vec = vec!(1, 2, 3, 4);
    let vec: &[int] = vec.as_slice(); //~ ERROR does not live long enough
    let init = match vec {
        [..init, _] => init,
        _ => fail!("b")
    };
    init
}

fn c() -> &[int] {
    let vec = vec!(1, 2, 3, 4);
    let vec: &[int] = vec.as_slice(); //~ ERROR does not live long enough
    let slice = match vec {
        [_, ..slice, _] => slice,
        _ => fail!("c")
    };
    slice
}

fn main() {}
