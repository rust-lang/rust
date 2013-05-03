// xfail-fast

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
use core::libc;
use core::sys;
use core::cast;
use std::arena::Arena;

struct Bcx<'self> {
    fcx: &'self Fcx<'self>
}

struct Fcx<'self> {
    arena: &'self mut Arena,
    ccx: &'self Ccx
}

struct Ccx {
    x: int
}

fn h<'r>(bcx : &'r mut Bcx<'r>) -> &'r mut Bcx<'r> {
    // XXX: Arena has a bad interface here; it should return mutable pointers.
    // But this patch is too big to roll that in.
    unsafe {
        cast::transmute(bcx.fcx.arena.alloc(|| Bcx { fcx: bcx.fcx }))
    }
}

fn g(fcx: &mut Fcx) {
    let mut bcx = Bcx { fcx: fcx };
    h(&mut bcx);
}

fn f(ccx: &mut Ccx) {
    let mut a = Arena();
    let mut fcx = Fcx { arena: &mut a, ccx: ccx };
    return g(&mut fcx);
}

pub fn main() {
    let mut ccx = Ccx { x: 0 };
    f(&mut ccx);
}

