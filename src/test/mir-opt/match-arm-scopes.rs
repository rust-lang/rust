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
//     switchInt((_2.0: bool)) -> [false: bb2, otherwise: bb3];
// }
// bb1 (cleanup): {
//     resume;
// }
// bb2: {                               // pre-binding for arm 1 first pattern
//     falseEdges -> [real: bb9, imaginary: bb4];
// }
// bb3: {
//     switchInt((_2.1: bool)) -> [false: bb4, otherwise: bb5];
// }
// bb4: {                               // pre-binding for arm 1 second pattern
//     falseEdges -> [real: bb18, imaginary: bb6];
// }
// bb5: {
//     switchInt((_2.0: bool)) -> [false: bb7, otherwise: bb6];
// }
// bb6: {                               // pre-binding for arm 2 first pattern
//     falseEdges -> [real: bb26, imaginary: bb7];
// }
// bb7: {                               // bindings for arm 2 - second pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb25;
// }
// bb8: {                               // arm 1
//     _0 = const 1i32;
//     drop(_7) -> [return: bb24, unwind: bb14];
// }
// bb9: {                               // guard - first time
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
//     switchInt(_10) -> [false: bb11, otherwise: bb10];
// }
// bb10: {
//     falseEdges -> [real: bb12, imaginary: bb11];
// }
// bb11: {                              // `else` block - first time
//     _9 = (*_6);
//     StorageDead(_10);
//     switchInt(move _9) -> [false: bb17, otherwise: bb16];
// }
// bb12: {                              // `return 3` - first time
//     _0 = const 3i32;
//     StorageDead(_10);
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb15;
// }
// bb13: {
//     return;
// }
// bb14 (cleanup): {
//     drop(_2) -> bb1;
// }
// bb15: {
//     drop(_2) -> [return: bb13, unwind: bb1];
// }
// bb16: {
//     StorageDead(_9);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     StorageLive(_5);
//     _5 = (_2.1: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb8;
// }
// bb17: {                              // guard otherwise case - first time
//     StorageDead(_9);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb3, imaginary: bb4];
// }
// bb18: {                              // guard - second time
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
//     switchInt(_13) -> [false: bb20, otherwise: bb19];
// }
// bb19: {
//     falseEdges -> [real: bb21, imaginary: bb20];
// }
// bb20: {                              // `else` block - second time
//     _12 = (*_6);
//     StorageDead(_13);
//     switchInt(move _12) -> [false: bb23, otherwise: bb22];
// }
// bb21: {
//     _0 = const 3i32;
//     StorageDead(_13);
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb15;
// }
// bb22: {                              // bindings for arm 1
//     StorageDead(_12);
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForGuardBinding, _6);
//     FakeRead(ForGuardBinding, _8);
//     StorageLive(_5);
//     _5 = (_2.0: bool);
//     StorageLive(_7);
//     _7 = move (_2.2: std::string::String);
//     goto -> bb8;
// }
// bb23: {                              // Guard otherwise case - second time
//     StorageDead(_12);
//     StorageDead(_8);
//     StorageDead(_6);
//     falseEdges -> [real: bb5, imaginary: bb6];
// }
// bb24: {                              // rest of arm 1
//     StorageDead(_7);
//     StorageDead(_5);
//     StorageDead(_8);
//     StorageDead(_6);
//     goto -> bb28;
// }
// bb25: {                              // arm 2
//     _0 = const 2i32;
//     drop(_16) -> [return: bb27, unwind: bb14];
// }
// bb26: {                              // bindings for arm 2 - first pattern
//     StorageLive(_15);
//     _15 = (_2.1: bool);
//     StorageLive(_16);
//     _16 = move (_2.2: std::string::String);
//     goto -> bb25;
// }

// bb27: {                              // rest of arm 2
//     StorageDead(_16);
//     StorageDead(_15);
//     goto -> bb28;
// }
// bb28: {
//     drop(_2) -> [return: bb13, unwind: bb1];
// }
// END rustc.complicated_match.SimplifyCfg-initial.after.mir
// START rustc.complicated_match.ElaborateDrops.after.mir
// let _16: std::string::String;      // No drop flags, which would come after this.
// scope 1 {
// END rustc.complicated_match.ElaborateDrops.after.mir
