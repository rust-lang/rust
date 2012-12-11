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
#[legacy_modes];

fn f(arg: {mut a: int}) {
    arg.a = 100;
}

fn main() {
    let x = {mut a: 10};
    f(x);
    assert x.a == 100;
    x.a = 20;
    f(copy x);
    assert x.a == 20;
}
