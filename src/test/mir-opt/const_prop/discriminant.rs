// compile-flags: -O

fn main() {
    let x = (if let Some(true) = Some(true) { 42 } else { 10 }) + 0;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = std::option::Option::<bool>::Some(const true,);
//      _4 = discriminant(_3);
//      switchInt(move _4) -> [1isize: bb3, otherwise: bb2];
//  }
//  bb1: {
//      _2 = const 42i32;
//      goto -> bb4;
//  }
//  bb2: {
//      _2 = const 10i32;
//      goto -> bb4;
//  }
//  bb3: {
//      switchInt(((_3 as Some).0: bool)) -> [false: bb2, otherwise: bb1];
//  }
//  bb4: {
//      _1 = Add(move _2, const 0i32);
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = const Scalar(0x01) : std::option::Option<bool>;
//      _4 = const 1isize;
//      switchInt(const 1isize) -> [1isize: bb3, otherwise: bb2];
//  }
//  bb1: {
//      _2 = const 42i32;
//      goto -> bb4;
//  }
//  bb2: {
//      _2 = const 10i32;
//      goto -> bb4;
//  }
//  bb3: {
//      switchInt(const true) -> [false: bb2, otherwise: bb1];
//  }
//  bb4: {
//      _1 = Add(move _2, const 0i32);
//      ...
//  }
// END rustc.main.ConstProp.after.mir
