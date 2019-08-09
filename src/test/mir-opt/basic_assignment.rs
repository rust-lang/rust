// this tests move up progration, which is not yet implemented

// Check codegen for assignments (`a = b`) where the left-hand-side is
// not yet initialized. Assignments tend to be absent in simple code,
// so subtle breakage in them can leave a quite hard-to-find trail of
// destruction.

fn main() {
    let nodrop_x = false;
    let nodrop_y;

    // Since boolean does not require drop, this can be a simple
    // assignment:
    nodrop_y = nodrop_x;

    let drop_x : Option<Box<u32>> = None;
    let drop_y;

    // Since the type of `drop_y` has drop, we generate a `replace`
    // terminator:
    drop_y = drop_x;
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-initial.after.mir
//    bb0: {
//        StorageLive(_1);
//        _1 = const false;
//        FakeRead(ForLet, _1);
//        StorageLive(_2);
//        StorageLive(_3);
//        _3 = _1;
//        _2 = move _3;
//        StorageDead(_3);
//        StorageLive(_4);
//        _4 = std::option::Option::<std::boxed::Box<u32>>::None;
//        FakeRead(ForLet, _4);
//        AscribeUserType(_4, o, UserTypeProjection { base: UserType(1), projs: [] });
//        StorageLive(_5);
//        StorageLive(_6);
//        _6 = move _4;
//        replace(_5 <- move _6) -> [return: bb2, unwind: bb5];
//    }
//    ...
//    bb2: {
//        drop(_6) -> [return: bb6, unwind: bb4];
//    }
//    ...
//    bb5 (cleanup): {
//        drop(_6) -> bb4;
//    }
// END rustc.main.SimplifyCfg-initial.after.mir
