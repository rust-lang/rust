fn try_identity(x: Result<u32, i32>) -> Result<u32, i32> {
    let y = x?;
    Ok(y)
}

fn main() {
    let _ = try_identity(Ok(0));
}

// END RUST SOURCE
// START rustc.try_identity.SimplifyArmIdentity.before.mir
// fn try_identity(_1: std::result::Result<u32, i32>) -> std::result::Result<u32, i32> {
//     debug x => _1;
//     let mut _0: std::result::Result<u32, i32>;
//     let _2: u32;
//     let mut _3: std::result::Result<u32, i32>;
//     let mut _4: std::result::Result<u32, i32>;
//     let mut _5: isize;
//     let _6: i32;
//     let mut _7: !;
//     let mut _8: i32;
//     let mut _9: i32;
//     let _10: u32;
//     let mut _11: u32;
//     scope 1 {
//         debug y => _10;
//     }
//     scope 2 {
//         debug err => _6;
//         scope 3 {
//             scope 7 {
//                 debug t => _6;
//             }
//             scope 8 {
//                 debug v => _6;
//                 let mut _12: i32;
//             }
//         }
//     }
//     scope 4 {
//         debug val => _10;
//         scope 5 {
//         }
//     }
//     scope 6 {
//         debug self => _1;
//     }
//     bb0: {
//         _5 = discriminant(_1);
//         switchInt(move _5) -> [0isize: bb4, 1isize: bb2, otherwise: bb1];
//     }
//     bb1: {
//         unreachable;
//     }
//     bb2: {
//         _6 = ((_1 as Err).0: i32);
//         ((_0 as Err).0: i32) = move _6;
//         discriminant(_0) = 1;
//         goto -> bb3;
//     }
//     bb3: {
//         return;
//     }
//     bb4: {
//         _10 = ((_1 as Ok).0: u32);
//         ((_0 as Ok).0: u32) = move _10;
//         discriminant(_0) = 0;
//         goto -> bb3;
//     }
// }
// END rustc.try_identity.SimplifyArmIdentity.before.mir

// START rustc.try_identity.SimplifyArmIdentity.after.mir
// fn try_identity(_1: std::result::Result<u32, i32>) -> std::result::Result<u32, i32> {
//     debug x => _1;
//     let mut _0: std::result::Result<u32, i32>;
//     let _2: u32;
//     let mut _3: std::result::Result<u32, i32>;
//     let mut _4: std::result::Result<u32, i32>;
//     let mut _5: isize;
//     let _6: i32;
//     let mut _7: !;
//     let mut _8: i32;
//     let mut _9: i32;
//     let _10: u32;
//     let mut _11: u32;
//     scope 1 {
//         debug y => _10;
//     }
//     scope 2 {
//         debug err => _6;
//         scope 3 {
//             scope 7 {
//                 debug t => _6;
//             }
//             scope 8 {
//                 debug v => _6;
//                 let mut _12: i32;
//             }
//         }
//     }
//     scope 4 {
//         debug val => _10;
//         scope 5 {
//         }
//     }
//     scope 6 {
//         debug self => _1;
//     }
//     bb0: {
//         _5 = discriminant(_1);
//         switchInt(move _5) -> [0isize: bb4, 1isize: bb2, otherwise: bb1];
//     }
//     bb1: {
//         unreachable;
//     }
//     bb2: {
//         _0 = move _1;
//         nop;
//         nop;
//         goto -> bb3;
//     }
//     bb3: {
//         return;
//     }
//     bb4: {
//         _0 = move _1;
//         nop;
//         nop;
//         goto -> bb3;
//     }
// }
// END rustc.try_identity.SimplifyArmIdentity.after.mir

// START rustc.try_identity.SimplifyBranchSame.after.mir
// fn try_identity(_1: std::result::Result<u32, i32>) -> std::result::Result<u32, i32> {
//     debug x => _1;
//     let mut _0: std::result::Result<u32, i32>;
//     let _2: u32;
//     let mut _3: std::result::Result<u32, i32>;
//     let mut _4: std::result::Result<u32, i32>;
//     let mut _5: isize;
//     let _6: i32;
//     let mut _7: !;
//     let mut _8: i32;
//     let mut _9: i32;
//     let _10: u32;
//     let mut _11: u32;
//     scope 1 {
//         debug y => _10;
//     }
//     scope 2 {
//         debug err => _6;
//         scope 3 {
//             scope 7 {
//                 debug t => _6;
//             }
//             scope 8 {
//                 debug v => _6;
//                 let mut _12: i32;
//             }
//         }
//     }
//     scope 4 {
//         debug val => _10;
//         scope 5 {
//         }
//     }
//     scope 6 {
//         debug self => _1;
//     }
//     bb0: {
//         _5 = discriminant(_1);
//         goto -> bb2;
//     }
//     bb1: {
//         return;
//     }
//     bb2: {
//         _0 = move _1;
//         nop;
//         nop;
//         goto -> bb1;
//     }
// }
// END rustc.try_identity.SimplifyBranchSame.after.mir

// START rustc.try_identity.SimplifyLocals.after.mir
// fn try_identity(_1: std::result::Result<u32, i32>) -> std::result::Result<u32, i32> {
//     debug x => _1;
//     let mut _0: std::result::Result<u32, i32>;
//     let mut _2: isize;
//     let _3: i32;
//     let _4: u32;
//     scope 1 {
//         debug y => _4;
//     }
//     scope 2 {
//         debug err => _3;
//         scope 3 {
//             scope 7 {
//                 debug t => _3;
//             }
//             scope 8 {
//                 debug v => _3;
//             }
//         }
//     }
//     scope 4 {
//         debug val => _4;
//         scope 5 {
//         }
//     }
//     scope 6 {
//         debug self => _1;
//     }
//     bb0: {
//         _2 = discriminant(_1);
//         _0 = move _1;
//         return;
//     }
// }
// END rustc.try_identity.SimplifyLocals.after.mir
