// compile-flags: -Z mir-opt-level=2

// Due to a bug in propagating scalar pairs the assertion below used to fail. In the expected
// outputs below, after ConstProp this is how _2 would look like with the bug:
//
//     _2 = (const Scalar(0x00) : (), const 0u8);
//
// Which has the wrong type.

fn encode(this: ((), u8, u8)) {
    assert!(this.2 == 0);
}

fn main() {
    encode(((), 0, 0));
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = ();
//      _2 = (move _3, const 0u8, const 0u8);
//      ...
//      _1 = const encode(move _2) -> bb1;
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = const Scalar(<ZST>) : ();
//      _2 = (move _3, const 0u8, const 0u8);
//      ...
//      _1 = const encode(move _2) -> bb1;
//      ...
//  }
// END rustc.main.ConstProp.after.mir
