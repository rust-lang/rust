// Test that StorageDead and Drops are generated properly for bindings in
// matches:
// * The MIR should only contain a single drop of `s` and `t`: at the end
//   of their respective arms.
// * StorageDead and StorageLive statements are correctly matched up on
//   non-unwind paths.
// * The visibility scopes of the match arms should be disjoint, and contain.
//   all of the bindings for that scope.
// * No drop flags are used.

fn complicated_match(cond: bool, items: (bool, bool, String)) -> i32 {
    match items {
        (false, a, s) | (a, false, s) if if cond { return 3 } else { a } => 1,
        (true, b, t) | (false, b, t) => 2,
    }
}

const CASES: &[(bool, bool, bool, i32)] = &[
    (false, false, false, 2),
    (false, false, true, 1),
    (false, true, false, 1),
    (false, true, true, 2),
    (true, false, false, 3),
    (true, false, true, 3),
    (true, true, false, 3),
    (true, true, true, 2),
];

fn main() {
    for &(cond, items_1, items_2, result) in CASES {
        assert_eq!(complicated_match(cond, (items_1, items_2, String::new())), result,);
    }
}

// END RUST SOURCE
// START rustc.complicated_match.SimplifyCfg-initial.after.mir
// let mut _0: i32;
// let mut _3: &bool;                   // Temp for fake borrow of `items.0`
// let mut _4: &bool;                   // Temp for fake borrow of `items.1`
// let _5: bool;                        // `a` in arm
// let _6: &bool;                       // `a` in guard
// let _7: std::string::String;         // `s` in arm
// let _8: &std::string::String;        // `s` in guard
// let mut _9: bool;                    // `if cond { return 3 } else { a }`
// let mut _10: bool;                   // `cond`
// let mut _11: !;                      // `return 3`
// let mut _12: bool;                   // `if cond { return 3 } else { a }`
// let mut _13: bool;                   // `cond`
// let mut _14: !;                      // `return 3`
// let _15: bool;                       // `b`
// let _16: std::string::String;        // `t`
// scope 1 {
//     debug a => _5;
//     debug a => _6;
//     debug s => _7;
//     debug s => _8;
// }
// scope 2 {
//     debug b => _15;
//     debug t => _16;
// }
// bb0: {
//     FakeRead(ForMatchedPlace, _2);
//     switchInt((_2.0: bool)) -> [false: bb1, otherwise: bb2];
// }
// bb1: {
//     falseEdges -> [real: bb8, imaginary: bb3];
// }
// bb2: {
//     switchInt((_2.1: bool)) -> [false: bb3, otherwise: bb4];
// }
// bb3: {
//     falseEdges -> [real: bb14, imaginary: bb5];
// }
// bb4: {
//     switchInt((_2.0: bool)) -> [false: bb6, otherwise: bb5];
// }
// bb5: {
//     falseEdges -> [real: bb22, imaginary: bb6];
// }
// bb6: {
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb21;
// }
// bb7: {
//     _0 = const 1i32;
//     drop(_7) -> [return: bb20, unwind: bb27];
// }
// bb8: {
//     StorageLive(_6);
//     _6 = &(_2.1: bool);
//     StorageLive(_8);
//     _8 = &(_2.2: std::string::String);
//     _3 = &shallow (_2.0: bool);
//     _4 = &shallow (_2.1: bool);
//     StorageLive(_9);
//     StorageLive(_10);
//     _10 = _1;
//     FakeRead(ForMatchedPlace, _10);
//     switchInt(_10) -> [false: bb10, otherwise: bb9];
// }
// bb9: {
//     falseEdges -> [real: bb11, imaginary: bb10];
// }
// bb10: {
//     _9 = (*_6);
//     StorageDead(_10);
//     switchInt(move _9) -> [false: bb13, otherwise: bb12];
// }
// bb11: {
//     _0 = const 3i32;
//     StorageDead(_10);
//     StorageDead(_9);
//     goto -> bb25;
// }
// bb12: {
//     StorageDead(_9);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     StorageLive(_5);
//     _5 = (_2.1: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb7;
// }
// bb13: {
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb2, imaginary: bb3];
// }
// bb14: {
//     StorageLive(_6);
//     _6 = &(_2.0: bool);
//     StorageLive(_8);
//     _8 = &(_2.2: std::string::String);
//     _3 = &shallow (_2.0: bool);
//     _4 = &shallow (_2.1: bool);
//     StorageLive(_12);
//     StorageLive(_13);
//     _13 = _1;
//     FakeRead(ForMatchedPlace, _13);
//     switchInt(_13) -> [false: bb16, otherwise: bb15];
// }
// bb15: {
//     falseEdges -> [real: bb17, imaginary: bb16];
// }
// bb16: {
//     _12 = (*_6);
//     StorageDead(_13);
//     switchInt(move _12) -> [false: bb19, otherwise: bb18];
// }
// bb17: {
//     _0 = const 3i32;
//     StorageDead(_13);
//     StorageDead(_12);
//     goto -> bb25;
// }
// bb18: {
//     StorageDead(_12);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     StorageLive(_5);
//     _5 = (_2.0: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb7;
// }
// bb19: {
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb4, imaginary: bb5];
// }
// bb20: {
//     StorageDead(_7);
//     StorageDead(_5);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb24;
// }
// bb21: {
//     _0 = const 2i32;
//     drop(_16) -> [return: bb23, unwind: bb27];
// }
// bb22: {
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb21;
// }
// bb23: {
//     StorageDead(_16);
//     StorageDead(_15);
//     goto -> bb24;
// }
// bb24: {
//     drop(_2) -> [return: bb26, unwind: bb28];
// }
// bb25: {
//     StorageDead(_8);
//     StorageDead(_6);
//     drop(_2) -> [return: bb26, unwind: bb28];
// }
// bb26: {
//     return;
// }
// bb27 (cleanup): {
//     drop(_2) -> bb28;
// }
// bb28 (cleanup): {
//     resume;
// }
// END rustc.complicated_match.SimplifyCfg-initial.after.mir
// START rustc.complicated_match.ElaborateDrops.after.mir
// let _16: std::string::String;      // No drop flags, which would come after this.
// scope 1 {
// END rustc.complicated_match.ElaborateDrops.after.mir
