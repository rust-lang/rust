fn test(x: u32) -> u32 {
    let y = x;
    y
}

fn main() {
    // Make sure the function actually gets instantiated.
    test(0);
}

// END RUST SOURCE
// START rustc.test.CopyPropagation.before.mir
//  bb0: {
//      ...
//      _2 = _1;
//      ...
//      _0 = _2;
//      ...
//      return;
//  }
// END rustc.test.CopyPropagation.before.mir
// START rustc.test.CopyPropagation.after.mir
//  bb0: {
//      ...
//      _0 = _1;
//      ...
//      return;
//  }
// END rustc.test.CopyPropagation.after.mir
