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
//        switchInt(move _6) -> [false: bb4, otherwise: bb1];
//    }
//    bb1: {
//        _7 = Lt(_1, const 10i32);
//        switchInt(move _7) -> [false: bb4, otherwise: bb2];
//    }
//    bb2: {
//        falseEdges -> [real: bb9, imaginary: bb6];
//    }
//    bb3: {
//        _3 = const 3i32;
//        goto -> bb14;
//    }
//    bb4: {
//        _4 = Le(const 10i32, _1);
//        switchInt(move _4) -> [false: bb7, otherwise: bb5];
//    }
//    bb5: {
//        _5 = Le(_1, const 20i32);
//        switchInt(move _5) -> [false: bb7, otherwise: bb6];
//    }
//    bb6: {
//        falseEdges -> [real: bb12, imaginary: bb8];
//    }
//    bb7: {
//        switchInt(_1) -> [-1i32: bb8, otherwise: bb3];
//    }
//    bb8: {
//        falseEdges -> [real: bb13, imaginary: bb3];
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
//        falseEdges -> [real: bb3, imaginary: bb6];
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
