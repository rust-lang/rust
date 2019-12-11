// compile-flags: -Z mir-opt-level=2

// This used to ICE in const-prop

fn test(this: ((u8, u8),)) {
    assert!((this.0).0 == 1);
}

fn main() {
    test(((1, 2),));
}

// Important bit is parameter passing so we only check that below
// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = (const 1u8, const 2u8);
//      _2 = (move _3,);
//      ...
//      _1 = const test(move _2) -> bb1;
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = (const 1u8, const 2u8);
//      _2 = (move _3,);
//      ...
//      _1 = const test(move _2) -> bb1;
//      ...
//  }
// END rustc.main.ConstProp.after.mir
