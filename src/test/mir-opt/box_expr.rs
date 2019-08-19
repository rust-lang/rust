// ignore-wasm32-bare compiled with panic=abort by default

#![feature(box_syntax)]

fn main() {
    let x = box S::new();
    drop(x);
}

struct S;

impl S {
    fn new() -> Self { S }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("splat!");
    }
}

// END RUST SOURCE
// START rustc.main.ElaborateDrops.before.mir
//     let mut _0: ();
//     let _1: std::boxed::Box<S>;
//     let mut _2: std::boxed::Box<S>;
//     let _3: ();
//     let mut _4: std::boxed::Box<S>;
//     scope 1 {
//     }
//     bb0: {
//         StorageLive(_1);
//         StorageLive(_2);
//         _2 = Box(S);
//         (*_2) = const S::new() -> [return: bb2, unwind: bb3];
//     }
//
//     bb1 (cleanup): {
//         resume;
//     }
//
//     bb2: {
//         _1 = move _2;
//         drop(_2) -> bb4;
//     }
//
//     bb3 (cleanup): {
//         drop(_2) -> bb1;
//     }
//
//     bb4: {
//         StorageDead(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = move _1;
//         _3 = const std::mem::drop::<std::boxed::Box<S>>(move _4) -> [return: bb5, unwind: bb7];
//     }
//
//     bb5: {
//         drop(_4) -> [return: bb8, unwind: bb6];
//     }
//
//     bb6 (cleanup): {
//         drop(_1) -> bb1;
//     }
//
//     bb7 (cleanup): {
//         drop(_4) -> bb6;
//     }
//
//     bb8: {
//         StorageDead(_4);
//         StorageDead(_3);
//         _0 = ();
//         drop(_1) -> bb9;
//     }
//
//     bb9: {
//         StorageDead(_1);
//         return;
//     }
// }
// END rustc.main.ElaborateDrops.before.mir
