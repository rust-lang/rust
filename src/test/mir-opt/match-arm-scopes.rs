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
// }
// scope 2 {
// }
// bb0: {
//     FakeRead(ForMatchedPlace, _2);
//     switchInt((_2.0: bool)) -> [false: bb2, otherwise: bb5];
// }
// bb1 (cleanup): {
//     resume;
// }
// bb2: {
//     falseEdges -> [real: bb8, imaginary: bb3];
// }
// bb3: {
//     falseEdges -> [real: bb17, imaginary: bb4];
// }
// bb4: {
//     falseEdges -> [real: bb25, imaginary: bb26];
// }
// bb5: {
//     switchInt((_2.1: bool)) -> [false: bb3, otherwise: bb6];
// }
// bb6: {
//     switchInt((_2.0: bool)) -> [false: bb26, otherwise: bb4];
// }
// bb7: {                               // arm 1
//     _0 = const 1i32;
//     drop(_7) -> [return: bb23, unwind: bb13];
// }
// bb8: {                               // guard - first time
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
// bb10: {                              // `else` block - first time
//     _9 = (*_6);
//     StorageDead(_10);
//     switchInt(move _9) -> [false: bb16, otherwise: bb15];
// }
// bb11: {                              // `return 3` - first time
//     _0 = const 3i32;
//     StorageDead(_10);
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb14;
// }
// bb12: {
//     return;
// }
// bb13 (cleanup): {
//     drop(_2) -> bb1;
// }
// bb14: {
//     drop(_2) -> [return: bb12, unwind: bb1];
// }
// bb15: {
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
// bb16: {                              // guard otherwise case - first time
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb5, imaginary: bb3];
// }
// bb17: {                              // guard - second time
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
//     switchInt(_13) -> [false: bb19, otherwise: bb18];
// }
// bb18: {
//     falseEdges -> [real: bb20, imaginary: bb19];
// }
// bb19: {                              // `else` block - second time
//     _12 = (*_6);
//     StorageDead(_13);
//     switchInt(move _12) -> [false: bb22, otherwise: bb21];
// }
// bb20: {
//     _0 = const 3i32;
//     StorageDead(_13);
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb14;
// }
// bb21: {                              // bindings for arm 1
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
// bb22: {                              // Guard otherwise case - second time
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb6, imaginary: bb4];
// }
// bb23: {                              // rest of arm 1
//     StorageDead(_7);
//     StorageDead(_5);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb28;
// }
// bb24: {                              // arm 2
//     _0 = const 2i32;
//     drop(_16) -> [return: bb27, unwind: bb13];
// }
// bb25: {                              // bindings for arm 2 - first pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb24;
// }
// bb26: {                              // bindings for arm 2 - second pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb24;
// }
// bb27: {                              // rest of arm 2
//     StorageDead(_16);
//     StorageDead(_15);
//     goto -> bb28;
// }
// bb28: {
//     drop(_2) -> [return: bb12, unwind: bb1];
// }
// END rustc.complicated_match.SimplifyCfg-initial.after.mir
// START rustc.complicated_match.ElaborateDrops.after.mir
// let _16: std::string::String;      // No drop flags, which would come after this.
// scope 1 {
// END rustc.complicated_match.ElaborateDrops.after.mir
