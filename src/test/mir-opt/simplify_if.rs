fn main() {
    if false {
        println!("hello world!");
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyBranches-after-const-prop.before.mir
// bb0: {
//     ...
//     switchInt(const false) -> [false: bb3, otherwise: bb1];
// }
// END rustc.main.SimplifyBranches-after-const-prop.before.mir
// START rustc.main.SimplifyBranches-after-const-prop.after.mir
// bb0: {
//     ...
//     goto -> bb3;
// }
// END rustc.main.SimplifyBranches-after-const-prop.after.mir
