// compile-flags: -O

static FOO: u8 = 2;

fn main() {
    let x = FOO + FOO;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = const Scalar(alloc0+0) : &u8;
//      _2 = (*_3);
//      ...
//      _5 = const Scalar(alloc0+0) : &u8;
//      _4 = (*_5);
//      _1 = Add(move _2, move _4);
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _2 = const 2u8;
//      ...
//      _4 = const 2u8;
//      _1 = const 4u8;
//      ...
//  }
// END rustc.main.ConstProp.after.mir
