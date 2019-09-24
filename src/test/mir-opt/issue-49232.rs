// We must mark a variable whose initialization fails due to an
// abort statement as StorageDead.

fn main() {
    loop {
        let beacon = {
            match true {
                false => 4,
                true => break,
            }
        };
        drop(&beacon);
    }
}

// END RUST SOURCE
// START rustc.main.mir_map.0.mir
// fn main() -> (){
//     let mut _0: ();
//     let mut _1: ();
//     let _2: i32;
//     let mut _3: bool;
//     let mut _4: !;
//     let _5: ();
//     let mut _6: &i32;
//     scope 1 {
//     }
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb3, cleanup: bb4];
//     }
//     bb2: {
//         goto -> bb14;
//     }
//     bb3: {
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = const true;
//         FakeRead(ForMatchedPlace, _3);
//         switchInt(_3) -> [false: bb5, otherwise: bb6];
//     }
//     bb4 (cleanup): {
//         resume;
//     }
//     bb5: {
//         falseEdges -> [real: bb7, imaginary: bb6];
//     }
//     bb6: {
//         _0 = ();
//         goto -> bb8;
//     }
//     bb7: {
//         _2 = const 4i32;
//         goto -> bb12;
//     }
//     bb8: {
//         StorageDead(_3);
//         goto -> bb9;
//     }
//     bb9: {
//         StorageDead(_2);
//         goto -> bb2;
//     }
//     bb10: {
//         _4 = ();
//         unreachable;
//     }
//     bb11: {
//         goto -> bb12;
//     }
//     bb12: {
//         FakeRead(ForLet, _2);
//         StorageDead(_3);
//         StorageLive(_5);
//         StorageLive(_6);
//         _6 = &_2;
//         _5 = const std::mem::drop::<&i32>(move _6) -> [return: bb13, unwind: bb4];
//     }
//     bb13: {
//         StorageDead(_6);
//         StorageDead(_5);
//         _1 = ();
//         StorageDead(_2);
//         goto -> bb1;
//     }
//     bb14: {
//         return;
//     }
// }
// END rustc.main.mir_map.0.mir
