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
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print unique.ptr[0]->val
// gdb-check:$1 = 10

// gdb-command:print unique.ptr[1]->val
// gdb-check:$2 = 11

// gdb-command:print unique.ptr[2]->val
// gdb-check:$3 = 12

// gdb-command:print unique.ptr[3]->val
// gdb-check:$4 = 13

#![allow(unused_variable)]

use std::gc::{Gc, GC};

fn main() {

    let unique: Vec<Gc<i64>> = vec!(box(GC) 10, box(GC) 11, box(GC) 12, box(GC) 13);

    zzz();
}

fn zzz() {()}
