// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when we compile the static `XXX` into MIR, we do not
// generate `StorageStart` or `StorageEnd` statements.

// ignore-tidy-linelength

static XXX: &'static Foo = &Foo {
    tup: "hi",
    data: &[
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
        (0, 1), (0, 2), (0, 3),
    ]
};

#[derive(Debug)]
struct Foo {
    tup: &'static str,
    data: &'static [(u32, u32)]
}

fn main() {
    println!("{:?}", XXX);
}

// END RUST SOURCE
// START rustc.node4.mir_map.0.mir
//    bb0: {
//        _7 = (const 0u32, const 1u32);   // scope 0 at src/test/mir-opt/basic_assignment.rs:29:9: 29:15
//        _8 = (const 0u32, const 2u32);   // scope 0 at src/test/mir-opt/basic_assignment.rs:29:17: 29:23
//        _9 = (const 0u32, const 3u32);   // scope 0 at src/test/mir-opt/basic_assignment.rs:29:25: 29:31
//        _10 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:30:9: 30:15
//        _11 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:30:17: 30:23
//        _12 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:30:25: 30:31
//        _13 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:31:9: 31:15
//        _14 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:31:17: 31:23
//        _15 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:31:25: 31:31
//        _16 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:32:9: 32:15
//        _17 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:32:17: 32:23
//        _18 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:32:25: 32:31
//        _19 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:33:9: 33:15
//        _20 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:33:17: 33:23
//        _21 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:33:25: 33:31
//        _22 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:34:9: 34:15
//        _23 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:34:17: 34:23
//        _24 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:34:25: 34:31
//        _25 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:35:9: 35:15
//        _26 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:35:17: 35:23
//        _27 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:35:25: 35:31
//        _28 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:36:9: 36:15
//        _29 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:36:17: 36:23
//        _30 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:36:25: 36:31
//        _31 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:37:9: 37:15
//        _32 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:37:17: 37:23
//        _33 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:37:25: 37:31
//        _34 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:38:9: 38:15
//        _35 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:38:17: 38:23
//        _36 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:38:25: 38:31
//        _37 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:39:9: 39:15
//        _38 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:39:17: 39:23
//        _39 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:39:25: 39:31
//        _40 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:40:9: 40:15
//        _41 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:40:17: 40:23
//        _42 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:40:25: 40:31
//        _43 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:41:9: 41:15
//        _44 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:41:17: 41:23
//        _45 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:41:25: 41:31
//        _46 = (const 0u32, const 1u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:42:9: 42:15
//        _47 = (const 0u32, const 2u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:42:17: 42:23
//        _48 = (const 0u32, const 3u32);  // scope 0 at src/test/mir-opt/basic_assignment.rs:42:25: 42:31
//        _6 = [_7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48]; // scope 0 at src/test/mir-opt/basic_assignment.rs:28:12: 43:6
//        _5 = &_6;                        // scope 0 at src/test/mir-opt/basic_assignment.rs:28:11: 43:6
//        _4 = &(*_5);                     // scope 0 at src/test/mir-opt/basic_assignment.rs:28:11: 43:6
//        _3 = _4 as &'static [(u32, u32)] (Unsize); // scope 0 at src/test/mir-opt/basic_assignment.rs:28:11: 43:6
//        _2 = Foo { tup: const "hi", data: _3 }; // scope 0 at src/test/mir-opt/basic_assignment.rs:26:29: 44:2
//        _1 = &_2;                        // scope 0 at src/test/mir-opt/basic_assignment.rs:26:28: 44:2
//        _0 = &(*_1);                     // scope 0 at src/test/mir-opt/basic_assignment.rs:26:28: 44:2
//        return;                          // scope 0 at src/test/mir-opt/basic_assignment.rs:26:1: 44:3
//    }
// END rustc.node4.mir_map.0.mir
