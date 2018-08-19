// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(box_syntax)]
#![feature(slice_patterns)]

fn move_out_from_begin_and_end() {
    let a = [box 1, box 2];
    let [_, _x] = a;
    let [.., _y] = a; //[ast]~ ERROR [E0382]
                      //[mir]~^ ERROR [E0382]
}

fn move_out_by_const_index_and_subslice() {
    let a = [box 1, box 2];
    let [_x, _] = a;
    let [_y..] = a; //[ast]~ ERROR [E0382]
                    //[mir]~^ ERROR [E0382]
}

fn main() {}
