// compile-flags: -O

static FOO: u8 = 2;

fn main() {
    let x = FOO + FOO;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _2 = (FOO: u8);
//      ...
//      _3 = (FOO: u8);
//      _1 = Add(move _2, move _3);
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _2 = const 2u8;
//      ...
//      _3 = const 2u8;
//      _1 = Add(move _2, move _3);
//      ...
//  }
// END rustc.main.ConstProp.after.mir
