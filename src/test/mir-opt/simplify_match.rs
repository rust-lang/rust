fn main() {
    match { let x = false; x } {
        true => println!("hello world!"),
        false => {},
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyBranches-after-copy-prop.before.mir
// bb0: {
//     ...
//     switchInt(const false) -> [false: bb3, otherwise: bb1];
// }
// bb1: {
// END rustc.main.SimplifyBranches-after-copy-prop.before.mir
// START rustc.main.SimplifyBranches-after-copy-prop.after.mir
// bb0: {
//     ...
//     goto -> bb3;
// }
// bb1: {
// END rustc.main.SimplifyBranches-after-copy-prop.after.mir
