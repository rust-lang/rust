fn main() {
    (&[1u32, 2, 3] as &[u32])[1];
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _4 = &(promoted[0]: [u32; 3]);
//      _3 = _4;
//      _2 = move _3 as &[u32] (Pointer(Unsize));
//      ...
//      _6 = const 1usize;
//      _7 = Len((*_2));
//      _8 = Lt(_6, _7);
//      assert(move _8, "index out of bounds: the len is move _7 but the index is _6") -> bb1;
//  }
//  bb1: {
//      _1 = (*_2)[_6];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _4 = const Scalar(AllocId(0).0x0) : &[u32; 3];
//      _3 = const Scalar(AllocId(0).0x0) : &[u32; 3];
//      _2 = move _3 as &[u32] (Pointer(Unsize));
//      ...
//      _6 = const 1usize;
//      _7 = const 3usize;
//      _8 = const true;
//      assert(const true, "index out of bounds: the len is move _7 but the index is _6") -> bb1;
//  }
//  bb1: {
//      _1 = (*_2)[_6];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
