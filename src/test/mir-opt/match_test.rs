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
//        _4 = Le(const 0i32, _1);
//        switchInt(move _4) -> [false: bb10, otherwise: bb11];
//    }
//    bb1: {
//        _3 = const 0i32;
//        goto -> bb16;
//    }
//    bb2: {
//        _3 = const 1i32;
//        goto -> bb16;
//    }
//    bb3: {
//        _3 = const 2i32;
//        goto -> bb16;
//    }
//    bb4: {
//        _3 = const 3i32;
//        goto -> bb16;
//    }
//    bb5: {
//        falseEdges -> [real: bb12, imaginary: bb6];
//    }
//    bb6: {
//        falseEdges -> [real: bb2, imaginary: bb7];
//    }
//    bb7: {
//        falseEdges -> [real: bb3, imaginary: bb8];
//    }
//    bb8: {
//        falseEdges -> [real: bb4, imaginary: bb9];
//    }
//    bb9: {
//        unreachable;
//    }
//    bb10: {
//        _7 = Le(const 10i32, _1);
//        switchInt(move _7) -> [false: bb14, otherwise: bb15];
//    }
//    bb11: {
//        _5 = Lt(_1, const 10i32);
//        switchInt(move _5) -> [false: bb10, otherwise: bb5];
//    }
//    bb12: {
//        StorageLive(_6);
//        _6 = _2;
//        switchInt(move _6) -> [false: bb13, otherwise: bb1];
//    }
//    bb13: {
//        falseEdges -> [real: bb8, imaginary: bb6];
//    }
//    bb14: {
//        switchInt(_1) -> [-1i32: bb7, otherwise: bb8];
//    }
//    bb15: {
//        _8 = Le(_1, const 20i32);
//        switchInt(move _8) -> [false: bb14, otherwise: bb6];
//    }
//    bb16: {
//        StorageDead(_6);
//        ...
//        return;
//    }
// END rustc.main.SimplifyCfg-initial.after.mir
