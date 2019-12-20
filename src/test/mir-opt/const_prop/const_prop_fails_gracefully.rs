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
//      _2 = &raw const (*_3);
//      _1 = move _2 as usize (Misc);
//      ...
//      _5 = _1;
//      _4 = const read(move _5) -> bb1;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = const main::FOO;
//      _2 = &raw const (*_3);
//      _1 = move _2 as usize (Misc);
//      ...
//      _5 = _1;
//      _4 = const read(move _5) -> bb1;
//  }
// END rustc.main.ConstProp.after.mir
