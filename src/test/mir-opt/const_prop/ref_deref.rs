fn main() {
    *(&4);
}

// END RUST SOURCE
// START rustc.main.PromoteTemps.before.mir
// bb0: {
//     ...
//     _3 = const 4i32;
//     _2 = &_3;
//     _1 = (*_2);
//     ...
//}
// END rustc.main.PromoteTemps.before.mir
// START rustc.main.PromoteTemps.after.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = &(*_4);
//     _1 = (*_2);
//     ...
//}
// END rustc.main.PromoteTemps.after.mir
// START rustc.main.ConstProp.before.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = _4;
//     _1 = (*_2);
//     ...
//}
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
// bb0: {
//     ...
//     _4 = const main::promoted[0];
//     _2 = _4;
//     _1 = const 4i32;
//     ...
// }
// END rustc.main.ConstProp.after.mir
