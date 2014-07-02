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
    let mut vec = [box 1i, box 2, box 3];
    match vec {
        [box ref _a, _, _] => {
            vec[0] = box 4; //~ ERROR cannot assign
        }
    }
}

fn b() {
    let mut vec = vec!(box 1i, box 2, box 3);
    let vec: &mut [Box<int>] = vec.as_mut_slice();
    match vec {
        [.._b] => {
            vec[0] = box 4; //~ ERROR cannot assign
        }
    }
}

fn c() {
    let mut vec = vec!(box 1i, box 2, box 3);
    let vec: &mut [Box<int>] = vec.as_mut_slice();
    match vec {
        [_a,         //~ ERROR cannot move out
         .._b] => {  //~^ NOTE attempting to move value to here

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
    let mut vec = vec!(box 1i, box 2, box 3);
    let vec: &mut [Box<int>] = vec.as_mut_slice();
    match vec {
        [.._a,     //~ ERROR cannot move out
         _b] => {} //~ NOTE attempting to move value to here
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
}

fn e() {
    let mut vec = vec!(box 1i, box 2, box 3);
    let vec: &mut [Box<int>] = vec.as_mut_slice();
    match vec {
        [_a, _b, _c] => {}  //~ ERROR cannot move out
        //~^ NOTE attempting to move value to here
        //~^^ NOTE and here
        //~^^^ NOTE and here
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~^ NOTE attempting to move value to here
}

fn main() {}
