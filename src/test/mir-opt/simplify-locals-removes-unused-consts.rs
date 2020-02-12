// compile-flags: -C overflow-checks=no

fn use_zst(_: ((), ())) {}

struct Temp {
    x: u8,
}

fn use_u8(_: u8) {}

fn main() {
    let ((), ()) = ((), ());
    use_zst(((), ()));

    use_u8((Temp { x: 40 }).x + 2);
}

// END RUST SOURCE

// START rustc.main.SimplifyLocals.before.mir
// let mut _0: ();
// let mut _1: ((), ());
// let mut _2: ();
// let mut _3: ();
// let _4: ();
// let mut _5: ((), ());
// let mut _6: ();
// let mut _7: ();
// let _8: ();
// let mut _9: u8;
// let mut _10: u8;
// let mut _11: Temp;
// scope 1 {
// }
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = const ();
//   StorageLive(_3);
//   _3 = const ();
//   _1 = const {transmute(()): ((), ())};
//   StorageDead(_3);
//   StorageDead(_2);
//   StorageDead(_1);
//   StorageLive(_4);
//   StorageLive(_6);
//   _6 = const ();
//   StorageLive(_7);
//   _7 = const ();
//   StorageDead(_7);
//   StorageDead(_6);
//   _4 = const use_zst(const {transmute(()): ((), ())}) -> bb1;
// }
// bb1: {
//   StorageDead(_4);
//   StorageLive(_8);
//   StorageLive(_10);
//   StorageLive(_11);
//   _11 = const {transmute(0x28) : Temp};
//   _10 = const 40u8;
//   StorageDead(_10);
//   _8 = const use_u8(const 42u8) -> bb2;
// }
// bb2: {
//   StorageDead(_11);
//   StorageDead(_8);
//   return;
// }
// END rustc.main.SimplifyLocals.before.mir
// START rustc.main.SimplifyLocals.after.mir
// let mut _0: ();
// let _1: ();
// let _2: ();
// scope 1 {
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const use_zst(const {transmute(()): ((), ())}) -> bb1;
// }
// bb1: {
//   StorageDead(_1);
//   StorageLive(_2);
//   _2 = const use_u8(const 42u8) -> bb2;
// }
// bb2: {
//   StorageDead(_2);
//   return;
// }
// END rustc.main.SimplifyLocals.after.mir
