// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

// ignore-tidy-linelength

// Issue #21633: reject duplicate loop labels in function bodies.
//
// This is testing the generalization (to the whole function body)
// discussed here:
// https://internals.rust-lang.org/t/psa-rejecting-duplicate-loop-labels/1833

pub fn foo() {
    { 'fl: for _ in 0..10 { break; } }   //~ NOTE first declared here
    { 'fl: loop { break; } }             //~ WARN label name `'fl` shadows a label name that is already in scope
    //~^ NOTE lifetime 'fl already in scope
    { 'lf: loop { break; } }             //~ NOTE first declared here
    { 'lf: for _ in 0..10 { break; } }   //~ WARN label name `'lf` shadows a label name that is already in scope
    //~^ NOTE lifetime 'lf already in scope
    { 'wl: while 2 > 1 { break; } }      //~ NOTE first declared here
    { 'wl: loop { break; } }             //~ WARN label name `'wl` shadows a label name that is already in scope
    //~^ NOTE lifetime 'wl already in scope
    { 'lw: loop { break; } }             //~ NOTE first declared here
    { 'lw: while 2 > 1 { break; } }      //~ WARN label name `'lw` shadows a label name that is already in scope
    //~^ NOTE lifetime 'lw already in scope
    { 'fw: for _ in 0..10 { break; } }   //~ NOTE first declared here
    { 'fw: while 2 > 1 { break; } }      //~ WARN label name `'fw` shadows a label name that is already in scope
    //~^ NOTE lifetime 'fw already in scope
    { 'wf: while 2 > 1 { break; } }      //~ NOTE first declared here
    { 'wf: for _ in 0..10 { break; } }   //~ WARN label name `'wf` shadows a label name that is already in scope
    //~^ NOTE lifetime 'wf already in scope
    { 'tl: while let Some(_) = None::<i32> { break; } } //~ NOTE first declared here
    { 'tl: loop { break; } }             //~ WARN label name `'tl` shadows a label name that is already in scope
    //~^ NOTE lifetime 'tl already in scope
    { 'lt: loop { break; } }             //~ NOTE first declared here
    { 'lt: while let Some(_) = None::<i32> { break; } }
                                         //~^ WARN label name `'lt` shadows a label name that is already in scope
                                         //~| NOTE lifetime 'lt already in scope
}

#[rustc_error]
pub fn main() { //~ ERROR compilation successful
    foo();
}
