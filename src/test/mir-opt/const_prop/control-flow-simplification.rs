// compile-flags: -Zmir-opt-level=1

trait NeedsDrop:Sized{
    const NEEDS:bool=std::mem::needs_drop::<Self>();
}

impl<This> NeedsDrop for This{}

fn hello<T>(){
    if <bool>::NEEDS {
        panic!()
    }
}

pub fn main() {
    hello::<()>();
    hello::<Vec<()>>();
}

// END RUST SOURCE
// START rustc.hello.ConstProp.before.mir
// let mut _0: ();
// let mut _1: bool;
// let mut _2: !;
// bb0: {
//   StorageLive(_1);
//   _1 = const <bool as NeedsDrop>::NEEDS;
//   switchInt(_1) -> [false: bb1, otherwise: bb2];
// }
// bb1: {
//   _0 = ();
//   StorageDead(_1);
//   return;
// }
// bb2: {
//   StorageLive(_2);
//   const std::rt::begin_panic::<&str>(const "explicit panic");
// }
// END rustc.hello.ConstProp.before.mir
// START rustc.hello.ConstProp.after.mir
// let mut _0: ();
// let mut _1: bool;
// let mut _2: !;
// bb0: {
//   StorageLive(_1);
//   _1 = const false;
//   switchInt(const false) -> [false: bb1, otherwise: bb2];
// }
// bb1: {
//   _0 = ();
//   StorageDead(_1);
//   return;
// }
// bb2: {
//   StorageLive(_2);
//   const std::rt::begin_panic::<&str>(const "explicit panic");
// }
// END rustc.hello.ConstProp.after.mir
// START rustc.hello.PreCodegen.before.mir
// let mut _0: ();
// bb0: {
//   return;
// }
// END rustc.hello.PreCodegen.before.mir
