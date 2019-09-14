// compile-flags: -O

fn main() {
    let x: u32 = [42; 8][2] + 0;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = [const 42u32; 8];
//      ...
//      _4 = const 2usize;
//      _5 = const 8usize;
//      _6 = Lt(_4, _5);
//      assert(move _6, "index out of bounds: the len is move _5 but the index is _4") -> bb1;
//  }
//  bb1: {
//      _2 = _3[_4];
//      _1 = Add(move _2, const 0u32);
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _6 = const true;
//      assert(const true, "index out of bounds: the len is move _5 but the index is _4") -> bb1;
//  }
//  bb1: {
//      _2 = const 42u32;
//      _1 = Add(move _2, const 0u32);
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
