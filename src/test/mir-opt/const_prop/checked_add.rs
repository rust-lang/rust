// compile-flags: -C overflow-checks=on

fn main() {
    let x: u32 = 1 + 1;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _2 = CheckedAdd(const 1u32, const 1u32);
//      assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _2 = (const 2u32, const false);
//      assert(!const false, "attempt to add with overflow") -> bb1;
//  }
// END rustc.main.ConstProp.after.mir
