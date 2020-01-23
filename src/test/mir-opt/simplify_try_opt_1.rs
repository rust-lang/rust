// ignore-tidy-linelength
// compile-flags: -Zmir-opt-level=1

// Note: Applying the optimization to `?` requires MIR inlining,
// which is not run in `mir-opt-level=1`.

fn try_identity(x: Result<u64, i32>) -> Result<u64, i32> {
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

fn main() {
    let _ = try_identity(Ok(0));
}

// END RUST SOURCE
// START rustc.try_identity.SimplifyArmIdentity.before.mir
// fn  try_identity(_1: std::result::Result<u64, i32>) -> std::result::Result<u64, i32> {
//     ...
//     bb0: {
//         _2 = discriminant(_1);
//         switchInt(move _2) -> [0isize: bb2, 1isize: bb3, otherwise: bb1];
//     }
//     bb1: {
//         unreachable;
//     }
//     bb2: {
//         StorageLive(_3);
//         _3 = ((_1 as Ok).0: u64);
//         StorageLive(_4);
//         _4 = _3;
//         ((_0 as Ok).0: u64) = move _4;
//         discriminant(_0) = 0;
//         StorageDead(_4);
//         StorageDead(_3);
//         goto -> bb4;
//     }
//     bb3: {
//         StorageLive(_5);
//         _5 = ((_1 as Err).0: i32);
//         StorageLive(_6);
//         _6 = _5;
//         ((_0 as Err).0: i32) = move _6;
//         discriminant(_0) = 1;
//         StorageDead(_6);
//         StorageDead(_5);
//         goto -> bb4;
//     }
//     bb4: {
//         return;
//     }
// }
// END rustc.try_identity.SimplifyArmIdentity.before.mir


// START rustc.try_identity.SimplifyArmIdentity.after.mir
// fn  try_identity(_1: std::result::Result<u64, i32>) -> std::result::Result<u64, i32> {
//     ...
//     bb0: {
//         _2 = discriminant(_1);
//         switchInt(move _2) -> [0isize: bb2, 1isize: bb3, otherwise: bb1];
//     }
//     bb1: {
//         unreachable;
//     }
//     bb2: {
//         _0 = move _1;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         goto -> bb4;
//     }
//     bb3: {
//         _0 = move _1;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         goto -> bb4;
//     }
//     bb4: {
//         return;
//     }
// }
// END rustc.try_identity.SimplifyArmIdentity.after.mir

// START rustc.try_identity.SimplifyBranchSame.after.mir
// fn  try_identity(_1: std::result::Result<u64, i32>) -> std::result::Result<u64, i32> {
//     ...
//     bb0: {
//         _2 = discriminant(_1);
//         goto -> bb1;
//     }
//     bb1: {
//         _0 = move _1;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         nop;
//         goto -> bb2;
//     }
//     bb2: {
//         return;
//     }
// }
// END rustc.try_identity.SimplifyBranchSame.after.mir
