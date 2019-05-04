fn test() -> &'static [u32] {
    &[1, 2]
}

fn main() {
    let x = test()[0];
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb1: {
//      ...
//      _3 = const 0usize;
//      _4 = Len((*_2));
//      _5 = Lt(_3, _4);
//      assert(move _5, "index out of bounds: the len is move _4 but the index is _3") -> bb2;
//  }
//  bb2: {
//      _1 = (*_2)[_3];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = const 0usize;
//      _4 = Len((*_2));
//      _5 = Lt(_3, _4);
//      assert(move _5, "index out of bounds: the len is move _4 but the index is _3") -> bb2;
//  }
//  bb2: {
//      _1 = (*_2)[_3];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
