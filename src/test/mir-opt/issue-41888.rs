// check that we clear the "ADT master drop flag" even when there are
// no fields to be dropped.

fn main() {
    let e;
    if cond() {
        e = E::F(K);
        if let E::F(_k) = e {
            // older versions of rustc used to not clear the
            // drop flag for `e` in this path.
        }
    }
}

fn cond() -> bool { false }

struct K;

enum E {
    F(K),
    G(Box<E>)
}

// END RUST SOURCE
// fn main() -> () {
//     let mut _0: ();
//     scope 1 {
//         let _1: E; // `e`
//         scope 2 {
//             let _6: K;
//         }
//     }
//     let mut _2: bool;
//     let mut _3: ();
//     let mut _4: E;
//     let mut _5: K;
//     let mut _7: isize;
//     let mut _8: bool; // drop flag for `e`
//     let mut _9: bool;
//     let mut _10: bool;
//     let mut _11: isize;
//     let mut _12: isize;
//
//     bb0: {
//         _8 = const false;
//         _10 = const false;
//         _9 = const false;
//         StorageLive(_1);
//         StorageLive(_2);
//         _2 = const cond() -> [return: bb3, unwind: bb2];
//     }
//
//     bb1: {
//         resume;
//     }
//
//     bb2: {
//         goto -> bb1;
//     }
//
//     bb3: {
//         switchInt(_2) -> [0u8: bb5, otherwise: bb4];
//     }
//
//     bb4: {
//         StorageLive(_4);
//         StorageLive(_5);
//         _5 = K::{{constructor}};
//         _4 = E::F(_5,);
//         StorageDead(_5);
//         goto -> bb15;
//     }
//
//     bb5: {
//         _0 = ();
//         goto -> bb12;
//     }
//
//     bb6: {
//         goto -> bb2;
//     }
//
//     bb7: {
//         goto -> bb8;
//     }
//
//     bb8: {
//         StorageDead(_4);
//         _7 = discriminant(_1);
//         switchInt(_7) -> [0isize: bb10, otherwise: bb9];
//     }
//
//     bb9: {
//         _0 = ();
//         goto -> bb11;
//     }
//
//     bb10: {
//         StorageLive(_6);
//         _10 = const false;
//         _6 = ((_1 as F).0: K);
//         _0 = ();
//         goto -> bb11;
//     }
//
//     bb11: {
//         StorageDead(_6);
//         goto -> bb12;
//     }
//
//     bb12: {
//         StorageDead(_2);
//         goto -> bb22;
//     }
//
//     bb13: {
//         StorageDead(_1);
//         return;
//     }
//
//     bb14: {
//         _8 = const true;
//         _9 = const true;
//         _10 = const true;
//         _1 = _4;
//         goto -> bb6;
//     }
//
//     bb15: {
//         _8 = const true;
//         _9 = const true;
//         _10 = const true;
//         _1 = _4;
//         goto -> bb7;
//     }
//
//     bb16: {
//         _8 = const false; // clear the drop flag - must always be reached
//         goto -> bb13;
//     }
//
//     bb17: {
//         _8 = const false;
//         goto -> bb1;
//     }
//
//     bb18: {
//         goto -> bb17;
//     }
//
//     bb19: {
//         drop(_1) -> [return: bb16, unwind: bb17];
//     }
//
//     bb20: {
//         drop(_1) -> bb17;
//     }
//
//     bb21: {
//         _11 = discriminant(_1);
//         switchInt(_11) -> [0isize: bb16, otherwise: bb19];
//     }
//
//     bb22: {
//         switchInt(_8) -> [0u8: bb16, otherwise: bb21];
//     }
//
//     bb23: {
//         _12 = discriminant(_1);
//         switchInt(_12) -> [0isize: bb18, otherwise: bb20];
//     }
//
//     bb24: {
//         switchInt(_8) -> [0u8: bb17, otherwise: bb23];
//     }
// }
