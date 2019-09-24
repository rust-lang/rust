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
//        switchInt(move _6) -> [false: bb6, otherwise: bb5];
//    }
//    bb1: {
//        falseEdges -> [real: bb9, imaginary: bb2];
//    }
//    bb2: {
//        falseEdges -> [real: bb12, imaginary: bb3];
//    }
//    bb3: {
//        falseEdges -> [real: bb13, imaginary: bb4];
//    }
//    bb4: {
//        _3 = const 3i32;
//        goto -> bb14;
//    }
//    bb5: {
//        _7 = Lt(_1, const 10i32);
//        switchInt(move _7) -> [false: bb6, otherwise: bb1];
//    }
//    bb6: {
//        _4 = Le(const 10i32, _1);
//        switchInt(move _4) -> [false: bb8, otherwise: bb7];
//    }
//    bb7: {
//        _5 = Le(_1, const 20i32);
//        switchInt(move _5) -> [false: bb8, otherwise: bb2];
//    }
//    bb8: {
//        switchInt(_1) -> [-1i32: bb3, otherwise: bb4];
//    }
//    bb9: {
//        _8 = &shallow _1;
//        StorageLive(_9);
//        _9 = _2;
//        switchInt(move _9) -> [false: bb11, otherwise: bb10];
//    }
//    bb10: {
//        StorageDead(_9);
//        FakeRead(ForMatchGuard, _8);
//        _3 = const 0i32;
//        goto -> bb14;
//    }
//    bb11: {
//        StorageDead(_9);
//        falseEdges -> [real: bb4, imaginary: bb2];
//    }
//    bb12: {
//        _3 = const 1i32;
//        goto -> bb14;
//    }
//    bb13: {
//        _3 = const 2i32;
//        goto -> bb14;
//    }
//    bb14: {
//        StorageDead(_3);
//        _0 = ();
//        StorageDead(_2);
//        StorageDead(_1);
//        return;
//    }
// END rustc.main.SimplifyCfg-initial.after.mir
