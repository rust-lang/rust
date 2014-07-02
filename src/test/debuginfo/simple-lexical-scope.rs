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
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$4 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$6 = 10
// gdb-command:continue

// gdb-command:finish
// gdb-command:print x
// gdb-check:$7 = false
// gdb-command:continue


fn main() {
    let x = false;

    zzz();
    sentinel();

    {
        zzz();
        sentinel();

        let x = 10i;

        zzz();
        sentinel();

        {
            zzz();
            sentinel();

            let x = 10.5f64;

            zzz();
            sentinel();
        }

        zzz();
        sentinel();
    }

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
