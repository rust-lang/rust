// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 Broken because of LLVM bug: http://llvm.org/bugs/show_bug.cgi?id=16249

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run

// debugger:finish
// debugger:print shadowed
// check:$1 = 231
// debugger:print not_shadowed
// check:$2 = 232
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$3 = 233
// debugger:print not_shadowed
// check:$4 = 232
// debugger:print local_to_arm
// check:$5 = 234
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$6 = 236
// debugger:print not_shadowed
// check:$7 = 232
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$8 = 237
// debugger:print not_shadowed
// check:$9 = 232
// debugger:print local_to_arm
// check:$10 = 238
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$11 = 239
// debugger:print not_shadowed
// check:$12 = 232
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$13 = 241
// debugger:print not_shadowed
// check:$14 = 232
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$15 = 243
// debugger:print *local_to_arm
// check:$16 = 244
// debugger:continue

// debugger:finish
// debugger:print shadowed
// check:$17 = 231
// debugger:print not_shadowed
// check:$18 = 232
// debugger:continue

struct Struct {
    x: int,
    y: int
}

fn main() {

    let shadowed = 231;
    let not_shadowed = 232;

    zzz();
    sentinel();

    match (233, 234) {
        (shadowed, local_to_arm) => {

            zzz();
            sentinel();
        }
    }

    match (235, 236) {
        // with literal
        (235, shadowed) => {

            zzz();
            sentinel();
        }
        _ => {}
    }

    match Struct { x: 237, y: 238 } {
        Struct { x: shadowed, y: local_to_arm } => {

            zzz();
            sentinel();
        }
    }

    match Struct { x: 239, y: 240 } {
        // ignored field
        Struct { x: shadowed, _ } => {

            zzz();
            sentinel();
        }
    }

    match Struct { x: 241, y: 242 } {
        // with literal
        Struct { x: shadowed, y: 242 } => {

            zzz();
            sentinel();
        }
        _ => {}
    }

    match (243, 244) {
        (shadowed, ref local_to_arm) => {

            zzz();
            sentinel();
        }
    }

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
