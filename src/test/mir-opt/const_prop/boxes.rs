// compile-flags: -O
// ignore-emscripten compiled with panic=abort by default
// ignore-wasm32
// ignore-wasm64

#![feature(box_syntax)]

// Note: this test verifies that we, in fact, do not const prop `box`

fn main() {
    let x = *(box 42) + 0;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _4 = Box(i32);
//      (*_4) = const 42i32;
//      _3 = move _4;
//      ...
//      _2 = (*_3);
//      _1 = Add(move _2, const 0i32);
//      ...
//      drop(_3) -> [return: bb2, unwind: bb1];
//  }
//  bb1 (cleanup): {
//      resume;
//  }
//  bb2: {
//      ...
//      _0 = ();
//      ...
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _4 = Box(i32);
//      (*_4) = const 42i32;
//      _3 = move _4;
//      ...
//      _2 = (*_3);
//      _1 = Add(move _2, const 0i32);
//      ...
//      drop(_3) -> [return: bb2, unwind: bb1];
//  }
//  bb1 (cleanup): {
//      resume;
//  }
//  bb2: {
//      ...
//      _0 = ();
//      ...
//  }
// END rustc.main.ConstProp.after.mir
