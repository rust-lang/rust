// ignore-wasm32-bare compiled with panic=abort by default

// Ensure that there are no drop terminators in `unwrap<T>` (except the one along the cleanup
// path).

fn unwrap<T>(opt: Option<T>) -> T {
    match opt {
        Some(x) => x,
        None => panic!(),
    }
}

fn main() {
    let _ = unwrap(Some(1i32));
}

// END RUST SOURCE
// START rustc.unwrap.SimplifyCfg-elaborate-drops.after.mir
// fn unwrap(_1: std::option::Option<T>) -> T {
//     ...
//     bb0: {
//         ...
//         switchInt(move _2) -> [0isize: bb2, 1isize: bb4, otherwise: bb3];
//     }
//     bb1 (cleanup): {
//         resume;
//     }
//     bb2: {
//         ...
//         const std::rt::begin_panic::<&'static str>(const "explicit panic") -> bb5;
//     }
//     bb3: {
//         unreachable;
//     }
//     bb4: {
//         ...
//         return;
//     }
//     bb5 (cleanup): {
//         drop(_1) -> bb1;
//     }
// }
// END rustc.unwrap.SimplifyCfg-elaborate-drops.after.mir
