// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

pub fn main() {
    let x: Box<_> = box 1;

    let v = (1, 2);

    match v {
        (1, _) if take(x) => (),
        (_, 2) if take(x) => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn take<T>(_: T) -> bool { false }
