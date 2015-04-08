// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Issue #21633: reject duplicate loop labels in function bodies.
// This is testing the exact cases that are in the issue description.

fn main() {
    'fl: for _ in 0..10 { break; } //~ NOTE shadowed label `'fl` declared here
    'fl: loop { break; }           //~ ERROR label name `'fl` shadows a label name that is already in scope

    'lf: loop { break; }           //~ NOTE shadowed label `'lf` declared here
    'lf: for _ in 0..10 { break; } //~ ERROR label name `'lf` shadows a label name that is already in scope

    'wl: while 2 > 1 { break; }    //~ NOTE shadowed label `'wl` declared here
    'wl: loop { break; }           //~ ERROR label name `'wl` shadows a label name that is already in scope

    'lw: loop { break; }           //~ NOTE shadowed label `'lw` declared here
    'lw: while 2 > 1 { break; }    //~ ERROR label name `'lw` shadows a label name that is already in scope

    'fw: for _ in 0..10 { break; } //~ NOTE shadowed label `'fw` declared here
    'fw: while 2 > 1 { break; }    //~ ERROR label name `'fw` shadows a label name that is already in scope

    'wf: while 2 > 1 { break; }    //~ NOTE shadowed label `'wf` declared here
    'wf: for _ in 0..10 { break; } //~ ERROR label name `'wf` shadows a label name that is already in scope

    'tl: while let Some(_) = None::<i32> { break; } //~ NOTE shadowed label `'tl` declared here
    'tl: loop { break; }           //~ ERROR label name `'tl` shadows a label name that is already in scope

    'lt: loop { break; }           //~ NOTE shadowed label `'lt` declared here
    'lt: while let Some(_) = None::<i32> { break; }
                                  //~^ ERROR label name `'lt` shadows a label name that is already in scope
}

// Note however that it is okay for the same label to be reuse in
// different methods of one impl, as illustrated here.

struct S;
impl S {
    fn m1(&self) { 'okay: loop { break 'okay; } }
    fn m2(&self) { 'okay: loop { break 'okay; } }
}
