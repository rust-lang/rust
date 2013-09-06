// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run

// BEFORE if
// debugger:finish
// debugger:print x
// check:$1 = 999
// debugger:print y
// check:$2 = -1
// debugger:continue

// AT BEGINNING of 'then' block
// debugger:finish
// debugger:print x
// check:$3 = 999
// debugger:print y
// check:$4 = -1
// debugger:continue

// AFTER 1st redeclaration of 'x'
// debugger:finish
// debugger:print x
// check:$5 = 1001
// debugger:print y
// check:$6 = -1
// debugger:continue

// AFTER 2st redeclaration of 'x'
// debugger:finish
// debugger:print x
// check:$7 = 1002
// debugger:print y
// check:$8 = 1003
// debugger:continue

// AFTER 1st if expression
// debugger:finish
// debugger:print x
// check:$9 = 999
// debugger:print y
// check:$10 = -1
// debugger:continue

// BEGINNING of else branch
// debugger:finish
// debugger:print x
// check:$11 = 999
// debugger:print y
// check:$12 = -1
// debugger:continue

// BEGINNING of else branch
// debugger:finish
// debugger:print x
// check:$13 = 1004
// debugger:print y
// check:$14 = 1005
// debugger:continue

// BEGINNING of else branch
// debugger:finish
// debugger:print x
// check:$15 = 999
// debugger:print y
// check:$16 = -1
// debugger:continue

use std::util;

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
        util::unreachable();
    }

    zzz();
    sentinel();

    if x > 1000 {
        util::unreachable();
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
