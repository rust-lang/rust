// Test that StorageDead and Drops are generated properly for bindings in
// matches:
// * The MIR should only contain a single drop of `s` and `t`: at the end
//   of their respective arms.
// * StorageDead and StorageLive statements are correctly matched up on
//   non-unwind paths.
// * The visibility scopes of the match arms should be disjoint, and contain.
//   all of the bindings for that scope.
// * No drop flags are used.

#![feature(nll, bind_by_move_pattern_guards)]

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
        assert_eq!(
            complicated_match(cond, (items_1, items_2, String::new())),
            result,
        );
    }
}

// END RUST SOURCE
// START rustc.complicated_match.SimplifyCfg-initial.after.mir
// let mut _0: i32;
// let mut _3: &bool;                   // Temp for fake borrow of `items.0`
// let mut _4: &bool;                   // Temp for fake borrow of `items.1`
// let _5: bool;                    // `a` in arm
// let _6: &bool;                   // `a` in guard
// let _7: std::string::String;     // `s` in arm
// let _8: &std::string::String;    // `s` in guard
// let mut _9: bool;                    // `if cond { return 3 } else { a }`
// let mut _10: bool;                   // `cond`
// let mut _11: !;                      // `return 3`
// let mut _12: bool;                   // `if cond { return 3 } else { a }`
// let mut _13: bool;                   // `cond`
// let mut _14: !;                      // `return 3`
// let _15: bool;                   // `b`
// let _16: std::string::String;    // `t`
// scope 1 {
// }
// scope 2 {
// }
// bb0: {
//     FakeRead(ForMatchedPlace, _2);
//     switchInt((_2.0: bool)) -> [false: bb2, otherwise: bb7];
// }
// bb1 (cleanup): {
//     resume;
// }
// bb2: {
//     falseEdges -> [real: bb10, imaginary: bb3];
// }
// bb3: {
//     falseEdges -> [real: bb21, imaginary: bb4];
// }
// bb4: {
//     falseEdges -> [real: bb31, imaginary: bb5];
// }
// bb5: {
//     falseEdges -> [real: bb32, imaginary: bb6];
// }
// bb6: {
//     unreachable;
// }
// bb7: {
//     switchInt((_2.1: bool)) -> [false: bb3, otherwise: bb8];
// }
// bb8: {
//     switchInt((_2.0: bool)) -> [false: bb5, otherwise: bb4];
// }
// bb9: {                               // arm 1
//     _0 = const 1i32;
//     drop(_7) -> [return: bb29, unwind: bb16];
// }
// bb10: {                              // guard - first time
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
//     switchInt(_10) -> [false: bb12, otherwise: bb11];
// }
// bb11: {
//     falseEdges -> [real: bb14, imaginary: bb12];
// }
// bb12: {
//     falseEdges -> [real: bb18, imaginary: bb13];
// }
// bb13: {
//     unreachable;
// }
// bb14: {                              // `return 3` - first time
//     _0 = const 3i32;
//     StorageDead(_10);
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb17;
// }
// bb15: {
//     return;
// }
// bb16 (cleanup): {
//     drop(_2) -> bb1;
// }
// bb17: {
//     drop(_2) -> [return: bb15, unwind: bb1];
// }
// bb18: {                              // `else` block - first time
//     _9 = (*_6);
//     StorageDead(_10);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     switchInt(move _9) -> [false: bb20, otherwise: bb19];
// }
// bb19: {
//     StorageDead(_9);
//     StorageLive(_5);
//     _5 = (_2.1: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb9;
// }
// bb20: {                              // guard otherwise case - first time
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb7, imaginary: bb3];
// }
// bb21: {                              // guard - second time
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
//     switchInt(_13) -> [false: bb23, otherwise: bb22];
// }
// bb22: {
//     falseEdges -> [real: bb25, imaginary: bb23];
// }
// bb23: {
//     falseEdges -> [real: bb26, imaginary: bb24];
// }
// bb24: {
//     unreachable;
// }
// bb25: {                              // `return 3` - second time
//     _0 = const 3i32;
//     StorageDead(_13);
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb17;
// }
// bb26: {                              // `else` block - second time
//     _12 = (*_6);
//     StorageDead(_13);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     switchInt(move _12) -> [false: bb28, otherwise: bb27];
// }
// bb27: {                              // Guard otherwise case - second time
//     StorageDead(_12);
//     StorageLive(_5);
//     _5 = (_2.0: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb9;
// }
// bb28: {                              // rest of arm 1
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb8, imaginary: bb4];
// }
// bb29: {
//     StorageDead(_7);
//     StorageDead(_5);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb34;
// }
// bb30: {                              // arm 2
//     _0 = const 2i32;
//     drop(_16) -> [return: bb33, unwind: bb16];
// }
// bb31: {                              // bindings for arm 2 - first pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb30;
// }
// bb32: {                              // bindings for arm 2 - first pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb30;
// }
// bb33: {                              // rest of arm 2
//     StorageDead(_16);
//     StorageDead(_15);
//     goto -> bb34;
// }
// bb34: {                              // end of match
//     drop(_2) -> [return: bb15, unwind: bb1];
// }
// END rustc.complicated_match.SimplifyCfg-initial.after.mir
// START rustc.complicated_match.ElaborateDrops.after.mir
// let _16: std::string::String;      // No drop flags, which would come after this.
// scope 1 {
// END rustc.complicated_match.ElaborateDrops.after.mir
