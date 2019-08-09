#[inline(never)]
fn read(_: usize) { }

fn main() {
    const FOO: &i32 = &1;
    let x = FOO as *const i32 as usize;
    read(x);
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = _4;
//      _2 = move _3 as *const i32 (Misc);
//      ...
//      _1 = move _2 as usize (Misc);
//      ...
//      _6 = _1;
//      _5 = const read(move _6) -> bb1;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _4 = const Scalar(AllocId(1).0x0) : &i32;
//      _3 = const Scalar(AllocId(1).0x0) : &i32;
//      _2 = const Scalar(AllocId(1).0x0) : *const i32;
//      ...
//      _1 = move _2 as usize (Misc);
//      ...
//      _6 = _1;
//      _5 = const read(move _6) -> bb1;
//  }
// END rustc.main.ConstProp.after.mir
