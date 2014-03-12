// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn a() {
    let mut vec = [~1, ~2, ~3];
    match vec {
        [~ref _a, _, _] => {
            vec[0] = ~4; //~ ERROR cannot assign
        }
    }
}

fn b() {
    let mut vec = vec!(~1, ~2, ~3);
    let vec: &mut [~int] = vec.as_mut_slice();
    match vec {
        [.._b] => {
            vec[0] = ~4; //~ ERROR cannot assign
        }
    }
}

fn c() {
    let mut vec = vec!(~1, ~2, ~3);
    let vec: &mut [~int] = vec.as_mut_slice();
    match vec {
        [_a, .._b] => {
            //~^ ERROR cannot move out

            // Note: `_a` is *moved* here, but `b` is borrowing,
            // hence illegal.
            //
            // See comment in middle/borrowck/gather_loans/mod.rs
            // in the case covering these sorts of vectors.
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
}

fn d() {
    let mut vec = vec!(~1, ~2, ~3);
    let vec: &mut [~int] = vec.as_mut_slice();
    match vec {
        [.._a, _b] => {
            //~^ ERROR cannot move out
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
}

fn e() {
    let mut vec = vec!(~1, ~2, ~3);
    let vec: &mut [~int] = vec.as_mut_slice();
    match vec {
        [_a, _b, _c] => {}  //~ ERROR cannot move out
        //~^ ERROR cannot move out
        //~^^ ERROR cannot move out
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
}

fn main() {}
