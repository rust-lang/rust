// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//xfail-test

extern mod std;

fn f(x : @{a:int, b:int}) {
    assert!((x.a == 10));
    assert!((x.b == 12));
}

pub fn main() {
    let z : @{a:int, b:int} = @{ a : 10, b : 12};
    let p = task::_spawn(bind f(z));
    task::join_id(p);
}
