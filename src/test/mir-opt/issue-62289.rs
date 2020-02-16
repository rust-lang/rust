// check that we don't forget to drop the Box if we early return before
// initializing it
// ignore-tidy-linelength
// ignore-wasm32-bare compiled with panic=abort by default

#![feature(box_syntax)]

fn test() -> Option<Box<u32>> {
    Some(box (None?))
}

fn main() {
    test();
}

// END RUST SOURCE
// START rustc.test.ElaborateDrops.before.mir
// fn test() -> std::option::Option<std::boxed::Box<u32>> {
//     ...
//     bb0: {
//         StorageLive(_1);
//         StorageLive(_2);
//         _2 = Box(u32);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = std::option::Option::<u32>::None;
//         _3 = const <std::option::Option<u32> as std::ops::Try>::into_result(move _4) -> [return: bb2, unwind: bb3];
//     }
//     bb1 (cleanup): {
//         resume;
//     }
//     bb2: {
//         StorageDead(_4);
//         _5 = discriminant(_3);
//         switchInt(move _5) -> [0isize: bb4, 1isize: bb6, otherwise: bb5];
//     }
//     bb3 (cleanup): {
//         drop(_2) -> bb1;
//     }
//     bb4: {
//         StorageLive(_10);
//         _10 = ((_3 as Ok).0: u32);
//         (*_2) = _10;
//         StorageDead(_10);
//         _1 = move _2;
//         drop(_2) -> [return: bb12, unwind: bb11];
//     }
//     bb5: {
//         unreachable;
//     }
//     bb6: {
//         StorageLive(_6);
//         _6 = ((_3 as Err).0: std::option::NoneError);
//         StorageLive(_8);
//         StorageLive(_9);
//         _9 = _6;
//         _8 = const <std::option::NoneError as std::convert::From<std::option::NoneError>>::from(move _9) -> [return: bb8, unwind: bb3];
//     }
//     bb7: {
//         return;
//     }
//     bb8: {
//         StorageDead(_9);
//         _0 = const <std::option::Option<std::boxed::Box<u32>> as std::ops::Try>::from_error(move _8) -> [return: bb9, unwind: bb3];
//     }
//     bb9: {
//         StorageDead(_8);
//         StorageDead(_6);
//         drop(_2) -> bb10;
//     }
//     bb10: {
//         StorageDead(_2);
//         StorageDead(_1);
//         StorageDead(_3);
//         goto -> bb7;
//     }
//     bb11 (cleanup): {
//         drop(_1) -> bb1;
//     }
//     bb12: {
//         StorageDead(_2);
//         _0 = std::option::Option::<std::boxed::Box<u32>>::Some(move _1,);
//         drop(_1) -> bb13;
//     }
//     bb13: {
//         StorageDead(_1);
//         StorageDead(_3);
//         goto -> bb7;
//     }
// }
// END rustc.test.ElaborateDrops.before.mir
