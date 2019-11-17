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
//         debug x => _1;
//     }
//     bb0: {
//         StorageLive(_1);
//         StorageLive(_2);
//         _2 = Box(S);
//         (*_2) = const S::new() -> [return: bb1, unwind: bb7];
//     }
//     bb1: {
//         _1 = move _2;
//         drop(_2) -> [return: bb2, unwind: bb6];
//     }
//     bb2: {
//         StorageDead(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = move _1;
//         _3 = const std::mem::drop::<std::boxed::Box<S>>(move _4) -> [return: bb3, unwind: bb5];
//     }
//     bb3: {
//         StorageDead(_4);
//         StorageDead(_3);
//         _0 = ();
//         drop(_1) -> bb4;
//     }
//     bb4: {
//         StorageDead(_1);
//         return;
//     }
//     bb5 (cleanup): {
//         drop(_4) -> bb6;
//     }
//     bb6 (cleanup): {
//         drop(_1) -> bb8;
//     }
//     bb7 (cleanup): {
//         drop(_2) -> bb8;
//     }
//     bb8 (cleanup): {
//         resume;
//     }
// }
// END rustc.main.ElaborateDrops.before.mir
