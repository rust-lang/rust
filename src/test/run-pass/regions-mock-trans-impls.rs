// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use libc, sys, cast;
use std::arena::Arena;

type bcx = {
    fcx: &fcx
};

type fcx = {
    arena: &Arena,
    ccx: &ccx
};

type ccx = {
    x: int
};

fn h(bcx : &r/bcx) -> &r/bcx {
    return bcx.fcx.arena.alloc(|| { fcx: bcx.fcx });
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    h(&bcx);
}

fn f(ccx : &ccx) {
    let a = Arena();
    let fcx = &{ arena: &a, ccx: ccx };
    return g(fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

