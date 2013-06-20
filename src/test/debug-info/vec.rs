// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

// compile-flags:-Z extra-debug-info
// debugger:set print pretty off
// debugger:break _zzz
// debugger:run
// debugger:finish
// debugger:print a
// check:$1 = {1, 2, 3}
// debugger:print b.vec[0]
// check:$2 = 4
// debugger:print c->boxed.data[1]
// check:$3 = 8
// debugger:print d->boxed.data[2]
// check:$4 = 12

fn main() {
    let a = [1, 2, 3];
    let b = &[4, 5, 6];
    let c = @[7, 8, 9];
    let d = ~[10, 11, 12];
    _zzz();
}

fn _zzz() {()}
