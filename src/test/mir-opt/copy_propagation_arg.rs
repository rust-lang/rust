// Check that CopyPropagation does not propagate an assignment to a function argument
// (doing so can break usages of the original argument value)

fn dummy(x: u8) -> u8 {
    x
}

fn foo(mut x: u8) {
    // calling `dummy` to make an use of `x` that copyprop cannot eliminate
    x = dummy(x); // this will assign a local to `x`
}

fn bar(mut x: u8) {
    dummy(x);
    x = 5;
}

fn baz(mut x: i32) {
    // self-assignment to a function argument should be eliminated
    x = x;
}

fn arg_src(mut x: i32) -> i32 {
    let y = x;
    x = 123; // Don't propagate this assignment to `y`
    y
}

fn main() {
    // Make sure the function actually gets instantiated.
    foo(0);
    bar(0);
    baz(0);
    arg_src(0);
}

// END RUST SOURCE
// START rustc.foo.CopyPropagation.before.mir
// bb0: {
//     ...
//     _3 = _1;
//     _2 = const dummy(move _3) -> bb1;
// }
// bb1: {
//     ...
//     _1 = move _2;
//     ...
// }
// END rustc.foo.CopyPropagation.before.mir
// START rustc.foo.CopyPropagation.after.mir
// bb0: {
//     ...
//     _3 = _1;
//     _2 = const dummy(move _3) -> bb1;
// }
// bb1: {
//     ...
//     _1 = move _2;
//     ...
// }
// END rustc.foo.CopyPropagation.after.mir
// START rustc.bar.CopyPropagation.before.mir
// bb0: {
//     StorageLive(_2);
//     StorageLive(_3);
//     _3 = _1;
//     _2 = const dummy(move _3) -> bb1;
// }
// bb1: {
//     StorageDead(_3);
//     StorageDead(_2);
//     _1 = const 5u8;
//     ...
//     return;
// }
// END rustc.bar.CopyPropagation.before.mir
// START rustc.bar.CopyPropagation.after.mir
// bb0: {
//     ...
//     _3 = _1;
//     _2 = const dummy(move _3) -> bb1;
// }
// bb1: {
//     ...
//     _1 = const 5u8;
//     ...
//     return;
// }
// END rustc.bar.CopyPropagation.after.mir
// START rustc.baz.CopyPropagation.before.mir
// bb0: {
//     StorageLive(_2);
//     _2 = _1;
//     _1 = move _2;
//     StorageDead(_2);
//     ...
//     return;
// }
// END rustc.baz.CopyPropagation.before.mir
// START rustc.baz.CopyPropagation.after.mir
// bb0: {
//     ...
//     _2 = _1;
//     _1 = move _2;
//     ...
//     return;
// }
// END rustc.baz.CopyPropagation.after.mir
// START rustc.arg_src.CopyPropagation.before.mir
// bb0: {
//      ...
//      _2 = _1;
//      ...
//      _1 = const 123i32;
//      ...
//      _0 = _2;
//      ...
//      return;
//  }
// END rustc.arg_src.CopyPropagation.before.mir
// START rustc.arg_src.CopyPropagation.after.mir
// bb0: {
//     ...
//     _2 = _1;
//     ...
//     _1 = const 123i32;
//     ...
//     _0 = _2;
//     ...
//     return;
// }
// END rustc.arg_src.CopyPropagation.after.mir
