// Test that the fake borrows for matches are removed after borrow checking.

// ignore-wasm32-bare

fn match_guard(x: Option<&&i32>, c: bool) -> i32 {
    match x {
        Some(0) if c => 0,
        _ => 1,
    }
}

fn main() {
    match_guard(None, true);
}

// END RUST SOURCE

// START rustc.match_guard.CleanupNonCodegenStatements.before.mir
// bb0: {
//     FakeRead(ForMatchedPlace, _1);
//     _3 = discriminant(_1);
//     switchInt(move _3) -> [1isize: bb3, otherwise: bb2];
// }
// bb1: {
//     goto -> bb4;
// }
// bb2: {
//     _0 = const 1i32;
//     goto -> bb7;
// }
// bb3: {
//     switchInt((*(*((_1 as Some).0: &'<empty> &'<empty> i32)))) -> [0i32: bb1, otherwise: bb2];
// }
// bb4: {
//     _4 = &shallow _1;
//     _5 = &shallow ((_1 as Some).0: &'<empty> &'<empty> i32);
//     _6 = &shallow (*((_1 as Some).0: &'<empty> &'<empty> i32));
//     _7 = &shallow (*(*((_1 as Some).0: &'<empty> &'<empty> i32)));
//     StorageLive(_8);
//     _8 = _2;
//     switchInt(move _8) -> [false: bb6, otherwise: bb5];
// }
// bb5: {
//     StorageDead(_8);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForMatchGuard, _5);
//     FakeRead(ForMatchGuard, _6);
//     FakeRead(ForMatchGuard, _7);
//     _0 = const 0i32;
//     goto -> bb7;
// }
// bb6: {
//     StorageDead(_8);
//     goto -> bb2;
// }
// bb7: {
//     return;
// }
// bb8 (cleanup): {
//     resume;
// }
// END rustc.match_guard.CleanupNonCodegenStatements.before.mir

// START rustc.match_guard.CleanupNonCodegenStatements.after.mir
// bb0: {
//     nop;
//     _3 = discriminant(_1);
//     switchInt(move _3) -> [1isize: bb3, otherwise: bb2];
// }
// bb1: {
//     goto -> bb4;
// }
// bb2: {
//     _0 = const 1i32;
//     goto -> bb7;
// }
// bb3: {
//     switchInt((*(*((_1 as Some).0: &'<empty> &'<empty> i32)))) -> [0i32: bb1, otherwise: bb2];
// }
// bb4: {
//     nop;
//     nop;
//     nop;
//     nop;
//     StorageLive(_8);
//     _8 = _2;
//     switchInt(move _8) -> [false: bb6, otherwise: bb5];
// }
// bb5: {
//     StorageDead(_8);
//     nop;
//     nop;
//     nop;
//     nop;
//     _0 = const 0i32;
//     goto -> bb7;
// }
// bb6: {
//     StorageDead(_8);
//     goto -> bb2;
// }
// bb7: {
//     return;
// }
// bb8 (cleanup): {
//     resume;
// }
// END rustc.match_guard.CleanupNonCodegenStatements.after.mir
