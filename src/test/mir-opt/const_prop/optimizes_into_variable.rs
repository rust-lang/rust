// compile-flags: -C overflow-checks=on

struct Point {
    x: u32,
    y: u32,
}

fn main() {
    let x = 2 + 2;
    let y = [0, 1, 2, 3, 4, 5][3];
    let z = (Point { x: 12, y: 42}).y;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
// let mut _0: ();
// let _1: i32;
// let mut _2: (i32, bool);
// let mut _4: [i32; 6];
// let _5: usize;
// let mut _6: usize;
// let mut _7: bool;
// let mut _9: Point;
// scope 1 {
//   debug x => _1;
//   let _3: i32;
//   scope 2 {
//     debug y => _3;
//     let _8: u32;
//     scope 3 {
//       debug z => _8;
//     }
//   }
// }
// bb0: {
//   StorageLive(_1);
//   _2 = CheckedAdd(const 2i32, const 2i32);
//   assert(!move (_2.1: bool), "attempt to add with overflow") -> bb1;
// }
// bb1: {
//   _1 = move (_2.0: i32);
//   StorageLive(_3);
//   StorageLive(_4);
//   _4 = [const 0i32, const 1i32, const 2i32, const 3i32, const 4i32, const 5i32];
//   StorageLive(_5);
//   _5 = const 3usize;
//   _6 = const 6usize;
//   _7 = Lt(_5, _6);
//   assert(move _7, "index out of bounds: the len is move _6 but the index is _5") -> bb2;
// }
// bb2: {
//   _3 = _4[_5];
//   StorageDead(_5);
//   StorageDead(_4);
//   StorageLive(_8);
//   StorageLive(_9);
//   _9 = Point { x: const 12u32, y: const 42u32 };
//   _8 = (_9.1: u32);
//   StorageDead(_9);
//   _0 = ();
//   StorageDead(_8);
//   StorageDead(_3);
//   StorageDead(_1);
//   return;
// }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
// let mut _0: ();
// let _1: i32;
// let mut _2: (i32, bool);
// let mut _4: [i32; 6];
// let _5: usize;
// let mut _6: usize;
// let mut _7: bool;
// let mut _9: Point;
// scope 1 {
//   debug x => _1;
//   let _3: i32;
//   scope 2 {
//     debug y => _3;
//     let _8: u32;
//     scope 3 {
//       debug z => _8;
//     }
//   }
// }
// bb0: {
//   StorageLive(_1);
//   _2 = (const 4i32, const false);
//   assert(!const false, "attempt to add with overflow") -> bb1;
// }
// bb1: {
//   _1 = const 4i32;
//   StorageLive(_3);
//   StorageLive(_4);
//   _4 = [const 0i32, const 1i32, const 2i32, const 3i32, const 4i32, const 5i32];
//   StorageLive(_5);
//   _5 = const 3usize;
//   _6 = const 6usize;
//   _7 = const true;
//   assert(const true, "index out of bounds: the len is move _6 but the index is _5") -> bb2;
// }
// bb2: {
//   _3 = const 3i32;
//   StorageDead(_5);
//   StorageDead(_4);
//   StorageLive(_8);
//   StorageLive(_9);
//   _9 = Point { x: const 12u32, y: const 42u32 };
//   _8 = const 42u32;
//   StorageDead(_9);
//   _0 = ();
//   StorageDead(_8);
//   StorageDead(_3);
//   StorageDead(_1);
//   return;
// }
// END rustc.main.ConstProp.after.mir
// START rustc.main.SimplifyLocals.after.mir
// let mut _0: ();
// let _1: i32;
// let mut _3: [i32; 6];
// scope 1 {
//   debug x => _1;
//   let _2: i32;
//   scope 2 {
//     debug y => _2;
//     let _4: u32;
//     scope 3 {
//       debug z => _4;
//     }
//   }
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const 4i32;
//   StorageLive(_2);
//   StorageLive(_3);
//   _3 = [const 0i32, const 1i32, const 2i32, const 3i32, const 4i32, const 5i32];
//   _2 = const 3i32;
//   StorageDead(_3);
//   StorageLive(_4);
//   _4 = const 42u32;
//   StorageDead(_4);
//   StorageDead(_2);
//   StorageDead(_1);
//   return;
// }
// END rustc.main.SimplifyLocals.after.mir
