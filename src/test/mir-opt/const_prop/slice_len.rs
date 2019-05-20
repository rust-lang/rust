static X: &[u32] = &[0, 1, 2];

fn main() {
    let x = X[1];
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _2 = const 1usize;
//      _3 = Len((*(X: &[u32])));
//      _4 = Lt(_2, _3);
//      assert(move _4, "index out of bounds: the len is move _3 but the index is _2") -> bb1;
//  }
//  bb1: {
//      _1 = (*(X: &[u32]))[_2];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _2 = const 1usize;
//      _3 = const 3usize;
//      _4 = const true;
//      assert(const true, "index out of bounds: the len is move _3 but the index is _2") -> bb1;
//  }
//  bb1: {
//      _1 = (*(X: &[u32]))[_2];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
