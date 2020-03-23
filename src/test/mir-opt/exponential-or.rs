// Test that simple or-patterns don't get expanded to exponentially large CFGs

// ignore-tidy-linelength

#![feature(or_patterns)]

fn match_tuple(x: (u32, bool, Option<i32>, u32)) -> u32 {
    match x {
        (y @ (1 | 4), true | false, Some(1 | 8) | None, z @ (6..=9 | 13..=16)) => y ^ z,
        _ => 0,
    }
}

fn main() {}

// END RUST SOURCE

// START rustc.match_tuple.SimplifyCfg-initial.after.mir
// scope 1 {
//     debug y => _7;
//     debug z => _8;
// }
// bb0: {
//     FakeRead(ForMatchedPlace, _1);
//     switchInt((_1.0: u32)) -> [1u32: bb2, 4u32: bb2, otherwise: bb1];
// }
// bb1: {
//     _0 = const 0u32;
//     goto -> bb10;
// }
// bb2: {
//     _2 = discriminant((_1.2: std::option::Option<i32>));
//     switchInt(move _2) -> [0isize: bb4, 1isize: bb3, otherwise: bb1];
// }
// bb3: {
//     switchInt((((_1.2: std::option::Option<i32>) as Some).0: i32)) -> [1i32: bb4, 8i32: bb4, otherwise: bb1];
// }
// bb4: {
//     _5 = Le(const 6u32, (_1.3: u32));
//     switchInt(move _5) -> [false: bb6, otherwise: bb5];
// }
// bb5: {
//     _6 = Le((_1.3: u32), const 9u32);
//     switchInt(move _6) -> [false: bb6, otherwise: bb8];
// }
// bb6: {
//     _3 = Le(const 13u32, (_1.3: u32));
//     switchInt(move _3) -> [false: bb1, otherwise: bb7];
// }
// bb7: {
//     _4 = Le((_1.3: u32), const 16u32);
//     switchInt(move _4) -> [false: bb1, otherwise: bb8];
// }
// bb8: {
//     falseEdges -> [real: bb9, imaginary: bb1];
// }
// bb9: {
//     StorageLive(_7);
//     _7 = (_1.0: u32);
//     StorageLive(_8);
//     _8 = (_1.3: u32);
//     StorageLive(_9);
//     _9 = _7;
//     StorageLive(_10);
//     _10 = _8;
//     _0 = BitXor(move _9, move _10);
//     StorageDead(_10);
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_7);
//     goto -> bb10;
// }
// bb10: {
//     return;
// }
// END rustc.match_tuple.SimplifyCfg-initial.after.mir
