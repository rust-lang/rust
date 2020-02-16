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
//   switchInt(move _3) -> [0isize: bb2, 1isize: bb3, otherwise: bb1];
// }
// bb1: {
//   StorageLive(_5);
//   _5 = const "C";
//   _1 = &(*_5);
//   StorageDead(_5);
//   goto -> bb4;
// }
// bb2: {
//   _1 = const "A(Empty)";
//   goto -> bb4;
// }
// bb3: {
//   StorageLive(_4);
//   _4 = const "B(Empty)";
//   _1 = &(*_4);
//   StorageDead(_4);
//   goto -> bb4;
// }
// bb4: {
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_6);
//   StorageLive(_7);
//   _7 = Test2::D;
//   _8 = discriminant(_7);
//   switchInt(move _8) -> [4isize: bb6, otherwise: bb5];
// }
// bb5: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb7;
// }
// bb6: {
//   _6 = const "D";
//   goto -> bb7;
// }
// bb7: {
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
//   switchInt(move _3) -> bb1;
// }
// bb1: {
//   StorageLive(_5);
//   _5 = const "C";
//   _1 = &(*_5);
//   StorageDead(_5);
//   goto -> bb4;
// }
// bb2: {
//   _1 = const "A(Empty)";
//   goto -> bb4;
// }
// bb3: {
//   StorageLive(_4);
//   _4 = const "B(Empty)";
//   _1 = &(*_4);
//   StorageDead(_4);
//   goto -> bb4;
// }
// bb4: {
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_6);
//   StorageLive(_7);
//   _7 = Test2::D;
//   _8 = discriminant(_7);
//   switchInt(move _8) -> [4isize: bb6, otherwise: bb5];
// }
// bb5: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb7;
// }
// bb6: {
//   _6 = const "D";
//   goto -> bb7;
// }
// bb7: {
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
//   switchInt(move _8) -> [4isize: bb2, otherwise: bb1];
// }
// bb1: {
//   StorageLive(_9);
//   _9 = const "E";
//   _6 = &(*_9);
//   StorageDead(_9);
//   goto -> bb3;
// }
// bb2: {
//   _6 = const "D";
//   goto -> bb3;
// }
// bb3: {
//   StorageDead(_7);
//   StorageDead(_6);
//   _0 = ();
//   return;
// }
// END rustc.main.SimplifyCfg-after-uninhabited-enum-branching.after.mir
