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
//     _2 = const get_bool(move _3) -> bb1;
// }
// bb1: {
//     StorageDead(_3);
//     switchInt(_2) -> [false: bb6, otherwise: bb2];
// }
// bb2: {
//      StorageLive(_4);
//      StorageLive(_5);
//      _5 = _1;
//      _4 = const get_bool(move _5) -> bb3;
// }
// bb3: {
//      StorageDead(_5);
//      switchInt(_4) -> [false: bb4, otherwise: bb5];
// }
// bb4: {
//      StorageDead(_4);
//      StorageDead(_2);
//      goto -> bb0;
// }
//  bb5: {
//      StorageDead(_4);
//      goto -> bb6;
//  }
//  bb6: {
//      StorageDead(_2);
//      return;
//  }
// END rustc.while_loop.PreCodegen.after.mir
