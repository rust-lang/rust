// Test that the fake borrows for matches are removed after borrow checking.

#![feature(nll)]

fn match_guard(x: Option<&&i32>) -> i32 {
    match x {
        Some(0) if true => 0,
        _ => 1,
    }
}

fn main() {
    match_guard(None);
}

// END RUST SOURCE

// START rustc.match_guard.CleanFakeReadsAndBorrows.before.mir
// bb0: {
//     FakeRead(ForMatchedPlace, _1);
//     _2 = discriminant(_1);
//     _3 = &shallow _1;
//     _4 = &shallow ((_1 as Some).0: &'<empty> &'<empty> i32);
//     _5 = &shallow (*((_1 as Some).0: &'<empty> &'<empty> i32));
//     _6 = &shallow (*(*((_1 as Some).0: &'<empty> &'<empty> i32)));
//     switchInt(move _2) -> [1isize: bb6, otherwise: bb4];
// }
// bb1: {
//     _0 = const 0i32;
//     goto -> bb9;
// }
// bb2: {
//     _0 = const 1i32;
//     goto -> bb9;
// }
// bb3: {
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForMatchGuard, _5);
//     FakeRead(ForMatchGuard, _6);
//     goto -> bb7;
// }
// bb4: {
//     FakeRead(ForMatchGuard, _3);
//     FakeRead(ForMatchGuard, _4);
//     FakeRead(ForMatchGuard, _5);
//     FakeRead(ForMatchGuard, _6);
//     goto -> bb2;
// }
// bb5: {
//     unreachable;
// }
// bb6: {
//     switchInt((*(*((_1 as Some).0: &'<empty> &'<empty> i32)))) -> [0i32: bb3, otherwise: bb4];
// }
// bb7: {
//     goto -> bb1;
// }
// bb8: {
//     goto -> bb4;
// }
// bb9: {
//     return;
// }
// bb10: {
//     resume;
// }
// END rustc.match_guard.CleanFakeReadsAndBorrows.before.mir

// START rustc.match_guard.CleanFakeReadsAndBorrows.after.mir
// bb0: {
//     nop;
//     _2 = discriminant(_1);
//     nop;
//     nop;
//     nop;
//     nop;
//     switchInt(move _2) -> [1isize: bb6, otherwise: bb4];
// }
// bb1: {
//     _0 = const 0i32;
//     goto -> bb9;
// }
// bb2: {
//     _0 = const 1i32;
//     goto -> bb9;
// }
// bb3: {
//     nop;
//     nop;
//     nop;
//     nop;
//     goto -> bb7;
// }
// bb4: {
//     nop;
//     nop;
//     nop;
//     nop;
//     goto -> bb2;
// }
// bb5: {
//     unreachable;
// }
// bb6: {
//     switchInt((*(*((_1 as Some).0: &'<empty> &'<empty> i32)))) -> [0i32: bb3, otherwise: bb4];
// }
// bb7: {
//     goto -> bb1;
// }
// bb8: {
//     goto -> bb4;
// }
// bb9: {
//     return;
// }
// bb10: {
//     resume;
// }
// END rustc.match_guard.CleanFakeReadsAndBorrows.after.mir
