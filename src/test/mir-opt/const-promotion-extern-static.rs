extern "C" {
    static X: i32;
}

static Y: i32 = 42;

static mut BAR: *const &'static i32 = [&Y].as_ptr();

static mut FOO: *const &'static i32 = [unsafe { &X }].as_ptr();

fn main() {}

// END RUST SOURCE
// START rustc.FOO.PromoteTemps.before.mir
// bb0: {
// ...
//     _5 = const Scalar(AllocId(1).0x0) : &i32;
//     _4 = &(*_5);
//     _3 = [move _4];
//     _2 = &_3;
//     _1 = move _2 as &[&'static i32] (Pointer(Unsize));
//     _0 = const core::slice::<impl [&'static i32]>::as_ptr(move _1) -> [return: bb2, unwind: bb1];
// }
// ...
// bb2: {
//     StorageDead(_5);
//     StorageDead(_3);
//     return;
// }
// END rustc.FOO.PromoteTemps.before.mir
// START rustc.BAR.PromoteTemps.before.mir
// bb0: {
// ...
//     _5 = const Scalar(AllocId(0).0x0) : &i32;
//     _4 = &(*_5);
//     _3 = [move _4];
//     _2 = &_3;
//     _1 = move _2 as &[&'static i32] (Pointer(Unsize));
//     _0 = const core::slice::<impl [&'static i32]>::as_ptr(move _1) -> [return: bb2, unwind: bb1];
// }
// ...
// bb2: {
//     StorageDead(_5);
//     StorageDead(_3);
//     return;
// }
// END rustc.BAR.PromoteTemps.before.mir
// START rustc.BAR.PromoteTemps.after.mir
// bb0: {
// ...
//     _2 = &(promoted[0]: [&'static i32; 1]);
//     _1 = move _2 as &[&'static i32] (Pointer(Unsize));
//     _0 = const core::slice::<impl [&'static i32]>::as_ptr(move _1) -> [return: bb2, unwind: bb1];
// }
// ...
// bb2: {
//     return;
// }
// END rustc.BAR.PromoteTemps.after.mir
// START rustc.FOO.PromoteTemps.after.mir
// bb0: {
// ...
//     _2 = &(promoted[0]: [&'static i32; 1]);
//     _1 = move _2 as &[&'static i32] (Pointer(Unsize));
//     _0 = const core::slice::<impl [&'static i32]>::as_ptr(move _1) -> [return: bb2, unwind: bb1];
// }
// ...
// bb2: {
//     return;
// }
// END rustc.FOO.PromoteTemps.after.mir
