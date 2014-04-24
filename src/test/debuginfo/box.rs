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

// compile-flags:-g
// gdb-command:set print pretty off
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish
// gdb-command:print *a
// gdb-check:$1 = 1
// gdb-command:print *b
// gdb-check:$2 = {2, 3.5}
// gdb-command:print c->val
// gdb-check:$3 = 4
// gdb-command:print d->val
// gdb-check:$4 = false

#![feature(managed_boxes)]
#![allow(unused_variable)]

fn main() {
    let a = box 1;
    let b = box() (2, 3.5);
    let c = @4;
    let d = @false;
    _zzz();
}

fn _zzz() {()}
