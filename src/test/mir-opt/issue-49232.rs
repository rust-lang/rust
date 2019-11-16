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
//         debug beacon => _2;
//     }
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb2, cleanup: bb11];
//     }
//     bb2: {
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = const true;
//         FakeRead(ForMatchedPlace, _3);
//         switchInt(_3) -> [false: bb3, otherwise: bb4];
//     }
//     bb3: {
//         falseEdges -> [real: bb5, imaginary: bb4];
//     }
//     bb4: {
//         _0 = ();
//         goto -> bb10;
//     }
//     bb5: {
//         _2 = const 4i32;
//          goto -> bb8;
//     }
//     bb6: {
//         _4 = ();
//         unreachable;
//      }
//     bb7: {
//          goto -> bb8;
//      }
//      bb8: {
//         FakeRead(ForLet, _2);
//         StorageDead(_3);
//         StorageLive(_5);
//          StorageLive(_6);
//         _6 = &_2;
//         _5 = const std::mem::drop::<&i32>(move _6) -> [return: bb9, unwind: bb11];
//     }
//     bb9: {
//         StorageDead(_6);
//         StorageDead(_5);
//         _1 = ();
//         StorageDead(_2);
//         goto -> bb1;
//     }
//     bb10: {
//         StorageDead(_3);
//         StorageDead(_2);
//         return;
//     }
//     bb11 (cleanup): {
//         resume;
//     }
// }
// END rustc.main.mir_map.0.mir
