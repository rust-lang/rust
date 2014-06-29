// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

#![feature(managed_boxes)]

// compile-flags:-g
// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print *ordinary_unique
// gdb-check:$1 = {-1, -2}

// gdb-command:print managed_within_unique->x
// gdb-check:$2 = -3

// gdb-command:print managed_within_unique->y->val
// gdb-check:$3 = -4

#![allow(unused_variable)]

use std::gc::{GC, Gc};

struct ContainsManaged {
    x: int,
    y: Gc<int>,
}

fn main() {
    let ordinary_unique = box() (-1i, -2i);

    let managed_within_unique = box ContainsManaged { x: -3, y: box(GC) -4i };

    zzz();
}

fn zzz() {()}
