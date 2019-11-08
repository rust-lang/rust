// compile-flags: -C overflow-checks=on

fn add() -> u32 {
    2 + 2
}

fn main() {
    add();
}

// END RUST SOURCE
// START rustc.add.ConstProp.before.mir
// fn add() -> u32 {
//     let mut _0: u32;
//     let mut _1: (u32, bool);
//     bb0: {
//         _1 = CheckedAdd(const 2u32, const 2u32);
//         assert(!move (_1.1: bool), "attempt to add with overflow") -> bb1;
//     }
//     bb1: {
//         _0 = move (_1.0: u32);
//         return;
//     }
//     bb2 (cleanup): {
//         resume;
//     }
// }
// END rustc.add.ConstProp.before.mir
// START rustc.add.ConstProp.after.mir
// fn add() -> u32 {
//     let mut _0: u32;
//     let mut _1: (u32, bool);
//     bb0: {
//         _1 = (const 4u32, const false);
//         assert(!const false, "attempt to add with overflow") -> bb1;
//     }
//     bb1: {
//         _0 = const 4u32;
//         return;
//     }
//     bb2 (cleanup): {
//         resume;
//     }
// }
// END rustc.add.ConstProp.after.mir
// START rustc.add.PreCodegen.before.mir
// fn add() -> u32 {
//     let mut _0: u32;
//     bb0: {
//         _0 = const 4u32;
//         return;
//     }
// }
// END rustc.add.PreCodegen.before.mir
