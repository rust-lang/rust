// check that we don't StorageDead booleans before they are used

fn main() {
    let mut should_break = false;
    loop {
        if should_break {
            break;
        }
        should_break = true;
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-initial.after.mir
//     bb0: {
//         StorageLive(_1);
//         _1 = const false;
//         FakeRead(ForLet, _1);
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb2, cleanup: bb6];
//     }
//     bb2: {
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = _1;
//         FakeRead(ForMatchedPlace, _4);
//         switchInt(_4) -> [false: bb4, otherwise: bb3];
//     }
//     ...
//     bb4: {
//         _3 = ();
//         StorageDead(_4);
//         StorageDead(_3);
//         _1 = const true;
//         _2 = ();
//         goto -> bb1;
//     }
//     bb5: {
//         _0 = ();
//         StorageDead(_4);
//         StorageDead(_3);
//         StorageDead(_1);
//         return;
//     }
//     bb6 (cleanup): {
//         resume;
//     }
// END rustc.main.SimplifyCfg-initial.after.mir
