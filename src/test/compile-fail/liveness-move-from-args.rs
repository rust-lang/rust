// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn take(-_x: int) { }

fn from_by_value_arg(++x: int) {
    take(move x);  //~ ERROR illegal move from argument `x`, which is not copy or move mode
}

fn from_by_ref_arg(&&x: int) {
    take(move x);  //~ ERROR illegal move from argument `x`, which is not copy or move mode
}

fn from_copy_arg(+x: int) {
    take(move x);
}

fn from_move_arg(-x: int) {
    take(move x);
}

fn main() {
}
