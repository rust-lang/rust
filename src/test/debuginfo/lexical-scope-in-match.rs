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
// gdb-command:print shadowed
// gdb-check:$1 = 231
// gdb-command:print not_shadowed
// gdb-check:$2 = 232
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$3 = 233
// gdb-command:print not_shadowed
// gdb-check:$4 = 232
// gdb-command:print local_to_arm
// gdb-check:$5 = 234
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$6 = 236
// gdb-command:print not_shadowed
// gdb-check:$7 = 232
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$8 = 237
// gdb-command:print not_shadowed
// gdb-check:$9 = 232
// gdb-command:print local_to_arm
// gdb-check:$10 = 238
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$11 = 239
// gdb-command:print not_shadowed
// gdb-check:$12 = 232
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$13 = 241
// gdb-command:print not_shadowed
// gdb-check:$14 = 232
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$15 = 243
// gdb-command:print *local_to_arm
// gdb-check:$16 = 244
// gdb-command:continue

// gdb-command:finish
// gdb-command:print shadowed
// gdb-check:$17 = 231
// gdb-command:print not_shadowed
// gdb-check:$18 = 232
// gdb-command:continue

struct Struct {
    x: int,
    y: int
}

fn main() {

    let shadowed = 231i;
    let not_shadowed = 232i;

    zzz();
    sentinel();

    match (233i, 234i) {
        (shadowed, local_to_arm) => {

            zzz();
            sentinel();
        }
    }

    match (235i, 236i) {
        // with literal
        (235, shadowed) => {

            zzz();
            sentinel();
        }
        _ => {}
    }

    match (Struct { x: 237, y: 238 }) {
        Struct { x: shadowed, y: local_to_arm } => {

            zzz();
            sentinel();
        }
    }

    match (Struct { x: 239, y: 240 }) {
        // ignored field
        Struct { x: shadowed, .. } => {

            zzz();
            sentinel();
        }
    }

    match (Struct { x: 241, y: 242 }) {
        // with literal
        Struct { x: shadowed, y: 242 } => {

            zzz();
            sentinel();
        }
        _ => {}
    }

    match (243i, 244i) {
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
