fn main() {
    let x: u32 = [0, 1, 2, 3][2];
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _2 = [const 0u32, const 1u32, const 2u32, const 3u32];
//      ...
//      _3 = const 2usize;
//      _4 = const 4usize;
//      _5 = Lt(_3, _4);
//      assert(move _5, "index out of bounds: the len is move _4 but the index is _3") -> bb1;
//  }
//  bb1: {
//      _1 = _2[_3];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _5 = const true;
//      assert(const true, "index out of bounds: the len is move _4 but the index is _3") -> bb1;
//  }
//  bb1: {
//      _1 = _2[_3];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
