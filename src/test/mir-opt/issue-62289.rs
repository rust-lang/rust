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
//         _3 = const <std::option::Option<u32> as std::ops::Try>::into_result(move _4) -> [return: bb1, unwind: bb12];
//     }
//     bb1: {
//         StorageDead(_4);
//         _5 = discriminant(_3);
//         switchInt(move _5) -> [0isize: bb6, 1isize: bb3, otherwise: bb2];
//     }
//     bb2: {
//         unreachable;
//     }
//     bb3: {
//         StorageLive(_6);
//         _6 = ((_3 as Err).0: std::option::NoneError);
//         StorageLive(_8);
//         StorageLive(_9);
//         _9 = _6;
//         _8 = const <std::option::NoneError as std::convert::From<std::option::NoneError>>::from(move _9) -> [return: bb4, unwind: bb12];
//     }
//     bb4: {
//         StorageDead(_9);
//         _0 = const <std::option::Option<std::boxed::Box<u32>> as std::ops::Try>::from_error(move _8) -> [return: bb5, unwind: bb12];
//     }
//     bb5: {
//         StorageDead(_8);
//         StorageDead(_6);
//         drop(_2) -> bb9;
//     }
//     bb6: {
//         StorageLive(_10);
//         _10 = ((_3 as Ok).0: u32);
//         (*_2) = _10;
//         StorageDead(_10);
//         _1 = move _2;
//         drop(_2) -> [return: bb7, unwind: bb11];
//     }
//     bb7: {
//         StorageDead(_2);
//         _0 = std::option::Option::<std::boxed::Box<u32>>::Some(move _1,);
//         drop(_1) -> bb8;
//     }
//     bb8: {
//         StorageDead(_1);
//         StorageDead(_3);
//         goto -> bb10;
//     }
//     bb9: {
//         StorageDead(_2);
//         StorageDead(_1);
//         StorageDead(_3);
//         goto -> bb10;
//     }
//     bb10: {
//         return;
//     }
//     bb11 (cleanup): {
//         drop(_1) -> bb13;
//     }
//     bb12 (cleanup): {
//         drop(_2) -> bb13;
//     }
//     bb13 (cleanup): {
//         resume;
//     }
// }
// END rustc.test.ElaborateDrops.before.mir
