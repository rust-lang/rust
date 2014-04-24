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
// gdb-check:$1 = false
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:print y
// gdb-check:$4 = true
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:print y
// gdb-check:$6 = 20
// gdb-command:continue

fn a_function(x: bool, y: bool) {
    zzz();
    sentinel();

    let x = 10;

    zzz();
    sentinel();

    let x = 10.5;
    let y = 20;

    zzz();
    sentinel();
}

fn main() {
    a_function(false, true);
}

fn zzz() {()}
fn sentinel() {()}
