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

// FIRST ITERATION
// gdb-command:finish
// gdb-command:print x
// gdb-check:$1 = 1
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$2 = -1
// gdb-command:continue

// SECOND ITERATION
// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 2
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$4 = -2
// gdb-command:continue

// THIRD ITERATION
// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = 3
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$6 = -3
// gdb-command:continue

// AFTER LOOP
// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = 1000000
// gdb-command:continue

fn main() {

    let range = [1, 2, 3];

    let x = 1000000; // wan meeeljen doollaars!

    for &x in range.iter() {
        zzz();
        sentinel();

        let x = -1 * x;

        zzz();
        sentinel();
    }

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
