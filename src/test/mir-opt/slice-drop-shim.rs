fn main() {
    std::ptr::drop_in_place::<[String]> as unsafe fn(_);
}

// END RUST SOURCE

// START rustc.ptr-real_drop_in_place.[std__string__String].AddMovesForPackedDrops.before.mir
// let mut _2: usize;
// let mut _3: usize;
// let mut _4: usize;
// let mut _5: &mut std::string::String;
// let mut _6: bool;
// let mut _7: &mut std::string::String;
// let mut _8: bool;
// let mut _9: *mut std::string::String;
// let mut _10: *mut std::string::String;
// let mut _11: &mut std::string::String;
// let mut _12: bool;
// let mut _13: &mut std::string::String;
// let mut _14: bool;
// let mut _15: *mut [std::string::String];
// bb0: {
//     goto -> bb15;
// }
// bb1: {
//     return;
// }
// bb2 (cleanup): {
//     resume;
// }
// bb3 (cleanup): {
//     _5 = &mut (*_1)[_4];
//     _4 = Add(move _4, const 1usize);
//     drop((*_5)) -> bb4;
// }
// bb4 (cleanup): {
//     _6 = Eq(_4, _3);
//     switchInt(move _6) -> [false: bb3, otherwise: bb2];
// }
// bb5: {
//     _7 = &mut (*_1)[_4];
//     _4 = Add(move _4, const 1usize);
//     drop((*_7)) -> [return: bb6, unwind: bb4];
// }
// bb6: {
//     _8 = Eq(_4, _3);
//     switchInt(move _8) -> [false: bb5, otherwise: bb1];
// }
// bb7: {
//     _4 = const 0usize;
//     goto -> bb6;
// }
// bb8: {
//     goto -> bb7;
// }
// bb9 (cleanup): {
//     _11 = &mut (*_9);
//     _9 = Offset(move _9, const 1usize);
//     drop((*_11)) -> bb10;
// }
// bb10 (cleanup): {
//     _12 = Eq(_9, _10);
//     switchInt(move _12) -> [false: bb9, otherwise: bb2];
// }
// bb11: {
//     _13 = &mut (*_9);
//     _9 = Offset(move _9, const 1usize);
//     drop((*_13)) -> [return: bb12, unwind: bb10];
// }
// bb12: {
//     _14 = Eq(_9, _10);
//     switchInt(move _14) -> [false: bb11, otherwise: bb1];
// }
// bb13: {
//     _15 = &mut (*_1);
//     _9 = move _15 as *mut std::string::String (Misc);
//     _10 = Offset(_9, move _3);
//     goto -> bb12;
// }
// bb14: {
//     goto -> bb13;
// }
// bb15: {
//     _2 = SizeOf(std::string::String);
//     _3 = Len((*_1));
//     switchInt(move _2) -> [0usize: bb8, otherwise: bb14];
// }
// END rustc.ptr-real_drop_in_place.[std__string__String].AddMovesForPackedDrops.before.mir
