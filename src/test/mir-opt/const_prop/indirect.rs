// compile-flags: -C overflow-checks=on

fn main() {
    let x = (2u32 as u8) + 1;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
// bb0: {
//     ...
//     _2 = const 2u32 as u8 (Misc);
//     _3 = CheckedAdd(move _2, const 1u8);
//     assert(!move (_3.1: bool), "attempt to add with overflow") -> bb1;
//}
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
// bb0: {
//     ...
//     _2 = const 2u8;
//     _3 = (const 3u8, const false);
//     assert(!const false, "attempt to add with overflow") -> bb1;
// }
// END rustc.main.ConstProp.after.mir
