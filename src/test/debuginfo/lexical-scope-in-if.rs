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

// BEFORE if
// gdb-command:finish
// gdb-command:print x
// gdb-check:$1 = 999
// gdb-command:print y
// gdb-check:$2 = -1
// gdb-command:continue

// AT BEGINNING of 'then' block
// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 999
// gdb-command:print y
// gdb-check:$4 = -1
// gdb-command:continue

// AFTER 1st redeclaration of 'x'
// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = 1001
// gdb-command:print y
// gdb-check:$6 = -1
// gdb-command:continue

// AFTER 2st redeclaration of 'x'
// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = 1002
// gdb-command:print y
// gdb-check:$8 = 1003
// gdb-command:continue

// AFTER 1st if expression
// gdb-command:finish
// gdb-command:print x
// gdb-check:$9 = 999
// gdb-command:print y
// gdb-check:$10 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:finish
// gdb-command:print x
// gdb-check:$11 = 999
// gdb-command:print y
// gdb-check:$12 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:finish
// gdb-command:print x
// gdb-check:$13 = 1004
// gdb-command:print y
// gdb-check:$14 = 1005
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:finish
// gdb-command:print x
// gdb-check:$15 = 999
// gdb-command:print y
// gdb-check:$16 = -1
// gdb-command:continue

fn main() {

    let x = 999;
    let y = -1;

    zzz();
    sentinel();

    if x < 1000 {
        zzz();
        sentinel();

        let x = 1001;

        zzz();
        sentinel();

        let x = 1002;
        let y = 1003;
        zzz();
        sentinel();
    } else {
        unreachable!();
    }

    zzz();
    sentinel();

    if x > 1000 {
        unreachable!();
    } else {
        zzz();
        sentinel();

        let x = 1004;
        let y = 1005;
        zzz();
        sentinel();
    }

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
