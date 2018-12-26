fn main() {
    if false {
        println!("hello world!");
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyBranches-initial.before.mir
// bb0: {
//     switchInt(const false) -> [false: bb3, otherwise: bb2];
// }
// END rustc.main.SimplifyBranches-initial.before.mir
// START rustc.main.SimplifyBranches-initial.after.mir
// bb0: {
//     goto -> bb3;
// }
// END rustc.main.SimplifyBranches-initial.after.mir
