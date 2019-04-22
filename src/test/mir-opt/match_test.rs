// Make sure redundant testing paths in `match` expressions are sorted out.

#![feature(exclusive_range_pattern)]

fn main() {
    let x = 3;
    let b = true;

    // When `(0..=10).contains(x) && !b`, we should jump to the last arm
    // without testing two other candidates.
    match x {
        0..10 if b => 0,
        10..=20 => 1,
        -1 => 2,
        _ => 3,
    };
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-initial.after.mir
//    bb0: {
//        ...
//        switchInt(move _4) -> [false: bb7, otherwise: bb8];
//    }
//    bb1: {
//        falseEdges -> [real: bb12, imaginary: bb2];
//    }
//    bb2: {
//        falseEdges -> [real: bb13, imaginary: bb3];
//    }
//    bb3: {
//        falseEdges -> [real: bb14, imaginary: bb4];
//    }
//    bb4: {
//        falseEdges -> [real: bb15, imaginary: bb5];
//    }
//    bb5: {
//        unreachable;
//    }
//    bb6: {
//        falseEdges -> [real: bb4, imaginary: bb2];
//    }
//    bb7: {
//        _6 = Le(const 10i32, _1);
//        switchInt(move _6) -> [false: bb9, otherwise: bb10];
//    }
//    bb8: {
//        _5 = Lt(_1, const 10i32);
//        switchInt(move _5) -> [false: bb7, otherwise: bb1];
//    }
//    bb9: {
//        switchInt(_1) -> [-1i32: bb3, otherwise: bb4];
//    }
//    bb10: {
//        _7 = Le(_1, const 20i32);
//        switchInt(move _7) -> [false: bb9, otherwise: bb2];
//    }
//    bb11: {
//        _3 = const 0i32;
//        goto -> bb16;
//    }
//    bb12: {
//        _8 = &shallow _1;
//        StorageLive(_9);
//        _9 = _2;
//        FakeRead(ForMatchGuard, _8);
//        switchInt(move _9) -> [false: bb6, otherwise: bb11];
//    }
//    bb13: {
//        _3 = const 1i32;
//        goto -> bb16;
//    }
//    bb14: {
//        _3 = const 2i32;
//        goto -> bb16;
//    }
//    bb15: {
//        _3 = const 3i32;
//        goto -> bb16;
//    }
//    bb16: {
//        StorageDead(_9);
//        _0 = ();
//        StorageDead(_2);
//        StorageDead(_1);
//        return;
//    }
// END rustc.main.SimplifyCfg-initial.after.mir
