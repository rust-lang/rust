fn main() {
    let x = 42u8 as u32;

    let y = 42u32 as u8;
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
// let mut _0: ();
// let _1: u32;
// scope 1 {
//   debug x => _1;
//   let _2: u8;
//   scope 2 {
//     debug y => _2;
//   }
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const 42u8 as u32 (Misc);
//   StorageLive(_2);
//   _2 = const 42u32 as u8 (Misc);
//   _0 = ();
//   StorageDead(_2);
//   StorageDead(_1);
//   return;
// }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
// let mut _0: ();
// let _1: u32;
// scope 1 {
//   debug x => _1;
//   let _2: u8;
//   scope 2 {
//     debug y => _2;
//   }
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const 42u32;
//   StorageLive(_2);
//   _2 = const 42u8;
//   _0 = ();
//   StorageDead(_2);
//   StorageDead(_1);
//   return;
// }
// END rustc.main.ConstProp.after.mir
