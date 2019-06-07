fn main() {
    let _ = main as usize as *const fn();
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = const main as fn() (Pointer(ReifyFnPointer));
//      _2 = move _3 as usize (Misc);
//      ...
//      _1 = move _2 as *const fn() (Misc);
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = const Scalar(AllocId(1).0x0) : fn();
//      _2 = move _3 as usize (Misc);
//      ...
//      _1 = const Scalar(AllocId(1).0x0) : *const fn();
//      ...
//  }
// END rustc.main.ConstProp.after.mir
