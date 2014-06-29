// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variable)]
#![allow(dead_assignment)]

fn cond() -> bool { fail!() }
fn link<'a>(v: &'a uint, w: &mut &'a uint) -> bool { *w = v; true }

fn separate_arms() {
    // Here both arms perform assignments, but only is illegal.

    let mut x = None;
    match x {
        None => {
            // It is ok to reassign x here, because there is in
            // fact no outstanding loan of x!
            x = Some(0i);
        }
        Some(ref _i) => {
            x = Some(1i); //~ ERROR cannot assign
        }
    }
    x.clone(); // just to prevent liveness warnings
}

fn guard() {
    // Here the guard performs a borrow. This borrow "infects" all
    // subsequent arms (but not the prior ones).

    let mut a = box 3u;
    let mut b = box 4u;
    let mut w = &*a;
    match 22i {
        _ if cond() => {
            b = box 5u;
        }

        _ if link(&*b, &mut w) => {
            b = box 6u; //~ ERROR cannot assign
        }

        _ => {
            b = box 7u; //~ ERROR cannot assign
        }
    }

    b = box 8; //~ ERROR cannot assign
}

fn main() {}
