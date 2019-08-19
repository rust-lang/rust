#[inline(never)]
fn foo(_: i32) { }

fn main() {
    match 1 {
        1 => foo(0),
        _ => foo(-1),
    }
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _1 = const 1i32;
//      switchInt(_1) -> [1i32: bb2, otherwise: bb1];
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      switchInt(const 1i32) -> [1i32: bb2, otherwise: bb1];
//  }
// END rustc.main.ConstProp.after.mir
// START rustc.main.SimplifyBranches-after-const-prop.before.mir
//  bb0: {
//      ...
//      _1 = const 1i32;
//      switchInt(const 1i32) -> [1i32: bb2, otherwise: bb1];
//  }
// END rustc.main.SimplifyBranches-after-const-prop.before.mir
// START rustc.main.SimplifyBranches-after-const-prop.after.mir
//  bb0: {
//      ...
//      _1 = const 1i32;
//      goto -> bb2;
//  }
// END rustc.main.SimplifyBranches-after-const-prop.after.mir
