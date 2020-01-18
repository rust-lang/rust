fn main() {
    *(&(4, 5).1); // This does not currently propagate (#67862)
}

// END RUST SOURCE
// START rustc.main.PromoteTemps.before.mir
// bb0: {
//     ...
//     _3 = (const 4i32, const 5i32);
//     _2 = &(_3.1: i32);
//     _1 = (*_2);
//     ...
//}
// END rustc.main.PromoteTemps.before.mir
// START rustc.main.PromoteTemps.after.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = &((*_4).1: i32);
//     _1 = (*_2);
//     ...
//}
// END rustc.main.PromoteTemps.after.mir
// START rustc.main.ConstProp.before.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = &((*_4).1: i32);
//     _1 = (*_2);
//     ...
//}
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = &((*_4).1: i32);
//     _1 = (*_2);
//     ...
// }
// END rustc.main.ConstProp.after.mir
