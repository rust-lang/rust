enum Empty { }

// test matching an enum with uninhabited variants
enum Test1 {
    A(Empty),
    B(Empty),
    C
}

// test an enum where the discriminants don't match the variant indexes
// (the optimization should do nothing here)
enum Test2 {
    D = 4,
    E = 5,
}

fn main() {
    match Test1::C {
        Test1::A(_) => "A(Empty)",
        Test1::B(_) => "B(Empty)",
        Test1::C => "C",
    };

    match Test2::D {
        Test2::D => "D",
        Test2::E => "E",
    };
}

// END RUST SOURCE
//
// START rustc.main.UninhabitedEnumBranching.before.mir
// let mut _0: ();
// let _1: &str;
// let mut _2: Test1;
// let mut _3: isize;
// let _4: &str;
// let _5: &str;
// let _6: &str;
// let mut _7: Test2;
// let mut _8: isize;
// let _9: &str;
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = Test1::C;
//   _3 = discriminant(_2);
//   switchInt(move _3) -> [0isize: bb3, 1isize: bb4, 2isize: bb1, otherwise: bb2];
// }
// bb1: {
//   StorageLive(_5);
//   _5 = const "C";
//   _1 = &(*_5);
//   StorageDead(_5);
//   goto -> bb5;
// }
// bb2: {
//   unreachable;
// }
// bb3: {
//   _1 = const "A(Empty)";
//   goto -> bb5;
// }
// bb4: {
//   StorageLive(_4);
//   _4 = const "B(Empty)";
//   _1 = &(*_4);
//   StorageDead(_4);
//   goto -> bb5;
// }
// bb5: {
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_6);
//   StorageLive(_7);
//   _7 = Test2::D;
//   _8 = discriminant(_7);
//   switchInt(move _8) -> [4isize: bb8, 5isize: bb6, otherwise: bb7];
// }
// bb6: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb9;
// }
// bb7: {
//   unreachable;
// }
// bb8: {
//   _6 = const "D";
//   goto -> bb9;
// }
// bb9: {
//   StorageDead(_7);
//   StorageDead(_6);
//   _0 = ();
//   return;
// }
// END rustc.main.UninhabitedEnumBranching.before.mir
// START rustc.main.UninhabitedEnumBranching.after.mir
// let mut _0: ();
// let _1: &str;
// let mut _2: Test1;
// let mut _3: isize;
// let _4: &str;
// let _5: &str;
// let _6: &str;
// let mut _7: Test2;
// let mut _8: isize;
// let _9: &str;
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = Test1::C;
//   _3 = discriminant(_2);
//   switchInt(move _3) -> [2isize: bb1, otherwise: bb2];
// }
// bb1: {
//   StorageLive(_5);
//   _5 = const "C";
//   _1 = &(*_5);
//   StorageDead(_5);
//   goto -> bb5;
// }
// bb2: {
//   unreachable;
// }
// bb3: {
//   _1 = const "A(Empty)";
//   goto -> bb5;
// }
// bb4: {
//   StorageLive(_4);
//   _4 = const "B(Empty)";
//   _1 = &(*_4);
//   StorageDead(_4);
//   goto -> bb5;
// }
// bb5: {
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_6);
//   StorageLive(_7);
//   _7 = Test2::D;
//   _8 = discriminant(_7);
//   switchInt(move _8) -> [4isize: bb8, 5isize: bb6, otherwise: bb7];
// }
// bb6: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb9;
// }
// bb7: {
//   unreachable;
// }
// bb8: {
//   _6 = const "D";
//   goto -> bb9;
// }
// bb9: {
//   StorageDead(_7);
//   StorageDead(_6);
//   _0 = ();
//   return;
// }
// END rustc.main.UninhabitedEnumBranching.after.mir
// START rustc.main.SimplifyCfg-after-uninhabited-enum-branching.after.mir
// let mut _0: ();
// let _1: &str;
// let mut _2: Test1;
// let mut _3: isize;
// let _4: &str;
// let _5: &str;
// let _6: &str;
// let mut _7: Test2;
// let mut _8: isize;
// let _9: &str;
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = Test1::C;
//   _3 = discriminant(_2);
//   switchInt(move _3) -> [2isize: bb1, otherwise: bb2];
// }
// bb1: {
//   StorageLive(_5);
//   _5 = const "C";
//   _1 = &(*_5);
//   StorageDead(_5);
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_6);
//   StorageLive(_7);
//   _7 = Test2::D;
//   _8 = discriminant(_7);
//   switchInt(move _8) -> [4isize: bb5, 5isize: bb3, otherwise: bb4];
// }
// bb2: {
//   unreachable;
// }
// bb3: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb6;
// }
// bb4: {
//   unreachable;
// }
// bb5: {
//   _6 = const "D";
//   goto -> bb6;
// }
// bb6: {
//   StorageDead(_7);
//   StorageDead(_6);
//   _0 = ();
//   return;
// }
// END rustc.main.SimplifyCfg-after-uninhabited-enum-branching.after.mir
