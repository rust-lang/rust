// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Note: the borrowck analysis is currently flow-insensitive.
// Therefore, some of these errors are marked as spurious and could be
// corrected by a simple change to the analysis.  The others are
// either genuine or would require more advanced changes.  The latter
// cases are noted.


fn borrow(_v: &int) {}
fn borrow_mut(_v: &mut int) {}
fn cond() -> bool { fail!() }
fn for_func(_f: || -> bool) { fail!() }
fn produce<T>() -> T { fail!(); }

fn inc(v: &mut Box<int>) {
    *v = box() (**v + 1);
}

fn pre_freeze_cond() {
    // In this instance, the freeze is conditional and starts before
    // the mut borrow.

    let mut v = box 3;
    let _w;
    if cond() {
        _w = &v;
    }
    borrow_mut(v); //~ ERROR cannot borrow
}

fn pre_freeze_else() {
    // In this instance, the freeze and mut borrow are on separate sides
    // of the if.

    let mut v = box 3;
    let _w;
    if cond() {
        _w = &v;
    } else {
        borrow_mut(v);
    }
}

fn main() {}
