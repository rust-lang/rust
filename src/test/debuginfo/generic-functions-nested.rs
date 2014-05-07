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
// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print x
// gdb-check:$1 = -1
// gdb-command:print y
// gdb-check:$2 = 1
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = -1
// gdb-command:print y
// gdb-check:$4 = 2.5
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = -2.5
// gdb-command:print y
// gdb-check:$6 = 1
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = -2.5
// gdb-command:print y
// gdb-check:$8 = 2.5
// gdb-command:continue

fn outer<TA: Clone>(a: TA) {
    inner(a.clone(), 1);
    inner(a.clone(), 2.5);

    fn inner<TX, TY>(x: TX, y: TY) {
        zzz();
    }
}

fn main() {
    outer(-1);
    outer(-2.5);
}

fn zzz() {()}
