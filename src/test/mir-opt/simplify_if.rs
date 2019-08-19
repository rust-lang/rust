fn main() {
    if false {
        println!("hello world!");
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyBranches-after-const-prop.before.mir
// bb0: {
//     ...
//     switchInt(const false) -> [false: bb1, otherwise: bb2];
// }
// END rustc.main.SimplifyBranches-after-const-prop.before.mir
// START rustc.main.SimplifyBranches-after-const-prop.after.mir
// bb0: {
//     ...
//     goto -> bb1;
// }
// END rustc.main.SimplifyBranches-after-const-prop.after.mir
