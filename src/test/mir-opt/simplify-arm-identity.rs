// Checks that `SimplifyArmIdentity` is not applied if enums have incompatible layouts.
// Regression test for issue #66856.
//
// compile-flags: -Zmir-opt-level=2

enum Src {
    Foo(u8),
    Bar,
}

enum Dst {
    Foo(u8),
}

fn main() {
    let e: Src = Src::Foo(0);
    let _: Dst = match e {
        Src::Foo(x) => Dst::Foo(x),
        Src::Bar => Dst::Foo(0),
    };
}

// END RUST SOURCE
// START rustc.main.SimplifyArmIdentity.before.mir
// fn main() -> () {
//     ...
//     bb0: {
//         StorageLive(_1);
//         ((_1 as Foo).0: u8) = const 0u8;
//         discriminant(_1) = 0;
//         StorageLive(_2);
//         _3 = discriminant(_1);
//         switchInt(move _3) -> [0isize: bb3, 1isize: bb1, otherwise: bb2];
//     }
//     bb1: {
//         ((_2 as Foo).0: u8) = const 0u8;
//         discriminant(_2) = 0;
//         goto -> bb4;
//     }
//     ...
//     bb3: {
//         _4 = ((_1 as Foo).0: u8);
//         ((_2 as Foo).0: u8) = move _4;
//         discriminant(_2) = 0;
//         goto -> bb4;
//     }
//     ...
// }
// END rustc.main.SimplifyArmIdentity.before.mir
// START rustc.main.SimplifyArmIdentity.after.mir
// fn main() -> () {
//     ...
//     bb0: {
//         StorageLive(_1);
//         ((_1 as Foo).0: u8) = const 0u8;
//         discriminant(_1) = 0;
//         StorageLive(_2);
//         _3 = discriminant(_1);
//         switchInt(move _3) -> [0isize: bb3, 1isize: bb1, otherwise: bb2];
//     }
//     bb1: {
//         ((_2 as Foo).0: u8) = const 0u8;
//         discriminant(_2) = 0;
//         goto -> bb4;
//     }
//     ...
//     bb3: {
//         _4 = ((_1 as Foo).0: u8);
//         ((_2 as Foo).0: u8) = move _4;
//         discriminant(_2) = 0;
//         goto -> bb4;
//     }
//     ...
// }
// END rustc.main.SimplifyArmIdentity.after.mir
