// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum arena = ();

type bcx = {
    fcx: &fcx
};

type fcx = {
    arena: &arena,
    ccx: &ccx
};

type ccx = {
    x: int
};

fn alloc(_bcx : &arena) -> &bcx {   
    unsafe {
        return cast::reinterpret_cast(
            &libc::malloc(sys::size_of::<bcx/&blk>() as libc::size_t));
    }
}

fn h(bcx : &bcx) -> &bcx {
    return alloc(bcx.fcx.arena);
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    let bcx2 = h(&bcx);
    unsafe {
        libc::free(cast::reinterpret_cast(&bcx2));
    }
}

fn f(ccx : &ccx) {
    let a = arena(());
    let fcx = { arena: &a, ccx: ccx };
    return g(&fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

