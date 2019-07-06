// Test that we correctly generate StorageDead statements for while loop
// conditions on all branches

fn get_bool(c: bool) -> bool {
    c
}

fn while_loop(c: bool) {
    while get_bool(c) {
        if get_bool(c) {
            break;
        }
    }
}

fn main() {
    while_loop(false);
}

// END RUST SOURCE

// START rustc.while_loop.PreCodegen.after.mir
// bb0: {
//     StorageLive(_2);
//     StorageLive(_3);
//     _3 = _1;
//     _2 = const get_bool(move _3) -> bb2;
// }
// bb1: {
//     return;
// }
// bb2: {
//     StorageDead(_3);
//     switchInt(move _2) -> [false: bb4, otherwise: bb3];
// }
// bb3: {
//     StorageDead(_2);
//     StorageLive(_4);
//     StorageLive(_5);
//     _5 = _1;
//     _4 = const get_bool(move _5) -> bb5;
// }
// bb4: {
//     StorageDead(_2);
//     goto -> bb1;
// }
// bb5: {
//     StorageDead(_5);
//     switchInt(_4) -> [false: bb6, otherwise: bb7];
// }
// bb6: {
//     StorageDead(_4);
//     goto -> bb0;
// }
// bb7: {
//     StorageDead(_4);
//     goto -> bb1;
// }
// END rustc.while_loop.PreCodegen.after.mir
