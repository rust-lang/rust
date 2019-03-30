fn main() {
    std::ptr::drop_in_place::<[String]> as unsafe fn(_);
}

// END RUST SOURCE

// START rustc.ptr-real_drop_in_place.[std__string__String].AddMovesForPackedDrops.before.mir
// let mut _2: usize;
// let mut _3: bool;
// let mut _4: usize;
// let mut _5: usize;
// let mut _6: &mut std::string::String;
// let mut _7: bool;
// let mut _8: &mut std::string::String;
// let mut _9: bool;
// let mut _10: *mut std::string::String;
// let mut _11: usize;
// let mut _12: *mut std::string::String;
// let mut _13: &mut std::string::String;
// let mut _14: bool;
// let mut _15: &mut std::string::String;
// let mut _16: bool;
// let mut _17: *mut [std::string::String];
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
//     _6 = &mut (*_1)[_4];
//     _4 = Add(_4, const 1usize);
//     drop((*_6)) -> bb4;
// }
// bb4 (cleanup): {
//     _7 = Eq(_4, _5);
//     switchInt(move _7) -> [false: bb3, otherwise: bb2];
// }
// bb5: {
//     _8 = &mut (*_1)[_4];
//     _4 = Add(_4, const 1usize);
//     drop((*_8)) -> [return: bb6, unwind: bb4];
// }
// bb6: {
//     _9 = Eq(_4, _5);
//     switchInt(move _9) -> [false: bb5, otherwise: bb1];
// }
// bb7: {
//     _5 = Len((*_1));
//     _4 = const 0usize;
//     goto -> bb6;
// }
// bb8: {
//     goto -> bb7;
// }
// bb9 (cleanup): {
//     _13 = &mut (*_10);
//     _10 = Offset(_10, const 1usize);
//     drop((*_13)) -> bb10;
// }
// bb10 (cleanup): {
//     _14 = Eq(_10, _12);
//     switchInt(move _14) -> [false: bb9, otherwise: bb2];
// }
// bb11: {
//     _15 = &mut (*_10);
//     _10 = Offset(_10, const 1usize);
//     drop((*_15)) -> [return: bb12, unwind: bb10];
// }
// bb12: {
//     _16 = Eq(_10, _12);
//     switchInt(move _16) -> [false: bb11, otherwise: bb1];
// }
// bb13: {
//     _11 = Len((*_1));
//     _17 = &mut (*_1);
//     _10 = move _17 as *mut std::string::String (Misc);
//     _12 = Offset(_10, move _11);
//     goto -> bb12;
// }
// bb14: {
//     goto -> bb13;
// }
// bb15: {
//     _2 = SizeOf(std::string::String);
//     _3 = Eq(move _2, const 0usize);
//     switchInt(move _3) -> [false: bb14, otherwise: bb8];
// }
// END rustc.ptr-real_drop_in_place.[std__string__String].AddMovesForPackedDrops.before.mir
