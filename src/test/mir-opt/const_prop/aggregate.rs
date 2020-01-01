// compile-flags: -O

fn main() {
    let x = (0, 1, 2).1 + 0;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = (const 0i32, const 1i32, const 2i32);
//      _2 = (_3.1: i32);
//      _1 = Add(move _2, const 0i32);
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = (const 0i32, const 1i32, const 2i32);
//      _2 = const 1i32;
//      _1 = const 1i32;
//      ...
//  }
// END rustc.main.ConstProp.after.mir
