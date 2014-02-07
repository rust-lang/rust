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
// debugger:rbreak zzz
// debugger:run

// FIRST ITERATION
// debugger:finish
// debugger:print x
// check:$1 = 0
// debugger:continue

// debugger:finish
// debugger:print x
// check:$2 = 1
// debugger:continue

// debugger:finish
// debugger:print x
// check:$3 = 101
// debugger:continue

// debugger:finish
// debugger:print x
// check:$4 = 101
// debugger:continue

// debugger:finish
// debugger:print x
// check:$5 = -987
// debugger:continue

// debugger:finish
// debugger:print x
// check:$6 = 101
// debugger:continue


// SECOND ITERATION
// debugger:finish
// debugger:print x
// check:$7 = 1
// debugger:continue

// debugger:finish
// debugger:print x
// check:$8 = 2
// debugger:continue

// debugger:finish
// debugger:print x
// check:$9 = 102
// debugger:continue

// debugger:finish
// debugger:print x
// check:$10 = 102
// debugger:continue

// debugger:finish
// debugger:print x
// check:$11 = -987
// debugger:continue

// debugger:finish
// debugger:print x
// check:$12 = 102
// debugger:continue

// debugger:finish
// debugger:print x
// check:$13 = 2
// debugger:continue

fn main() {

    let mut x = 0;

    loop {
        if x >= 2 {
            break;
        }

        zzz();
        sentinel();

        x += 1;
        zzz();
        sentinel();

        // Shadow x
        let x = x + 100;
        zzz();
        sentinel();

        // open scope within loop's top level scope
        {
            zzz();
            sentinel();

            let x = -987;

            zzz();
            sentinel();
        }

        // Check that we get the x before the inner scope again
        zzz();
        sentinel();
    }

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
