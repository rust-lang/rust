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

// === GDB TESTS ===================================================================================

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


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print shadowed
// lldb-check:[...]$0 = 231
// lldb-command:print not_shadowed
// lldb-check:[...]$1 = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$2 = 233
// lldb-command:print not_shadowed
// lldb-check:[...]$3 = 232
// lldb-command:print local_to_arm
// lldb-check:[...]$4 = 234
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$5 = 236
// lldb-command:print not_shadowed
// lldb-check:[...]$6 = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$7 = 237
// lldb-command:print not_shadowed
// lldb-check:[...]$8 = 232
// lldb-command:print local_to_arm
// lldb-check:[...]$9 = 238
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$10 = 239
// lldb-command:print not_shadowed
// lldb-check:[...]$11 = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$12 = 241
// lldb-command:print not_shadowed
// lldb-check:[...]$13 = 232
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$14 = 243
// lldb-command:print *local_to_arm
// lldb-check:[...]$15 = 244
// lldb-command:continue

// lldb-command:print shadowed
// lldb-check:[...]$16 = 231
// lldb-command:print not_shadowed
// lldb-check:[...]$17 = 232
// lldb-command:continue


struct Struct {
    x: int,
    y: int
}

fn main() {

    let shadowed = 231i;
    let not_shadowed = 232i;

    zzz(); // #break
    sentinel();

    match (233i, 234i) {
        (shadowed, local_to_arm) => {

            zzz(); // #break
            sentinel();
        }
    }

    match (235i, 236i) {
        // with literal
        (235, shadowed) => {

            zzz(); // #break
            sentinel();
        }
        _ => {}
    }

    match (Struct { x: 237, y: 238 }) {
        Struct { x: shadowed, y: local_to_arm } => {

            zzz(); // #break
            sentinel();
        }
    }

    match (Struct { x: 239, y: 240 }) {
        // ignored field
        Struct { x: shadowed, .. } => {

            zzz(); // #break
            sentinel();
        }
    }

    match (Struct { x: 241, y: 242 }) {
        // with literal
        Struct { x: shadowed, y: 242 } => {

            zzz(); // #break
            sentinel();
        }
        _ => {}
    }

    match (243i, 244i) {
        (shadowed, ref local_to_arm) => {

            zzz(); // #break
            sentinel();
        }
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
