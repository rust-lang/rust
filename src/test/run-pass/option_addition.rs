// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let foo = 1;
    let bar = 2;
    let foobar = foo + bar;

    let nope = optint(0) + optint(0);
    let somefoo = optint(foo) + optint(0);
    let somebar = optint(bar) + optint(0);
    let somefoobar = optint(foo) + optint(bar);

    match nope {
        None => (),
        Some(foo) => fail!(fmt!("expected None, but found %?", foo))
    }
    assert!(foo == somefoo.get());
    assert!(bar == somebar.get());
    assert!(foobar == somefoobar.get());
}

fn optint(in: int) -> Option<int> {
    if in == 0 {
        return None;
    }
    else {
        return Some(in);
    }
}
