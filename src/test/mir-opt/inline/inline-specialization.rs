#![feature(specialization)]

fn main() {
    let x = <Vec::<()> as Foo>::bar();
}

trait Foo {
    fn bar() -> u32;
}

impl<T> Foo for Vec<T> {
    #[inline(always)]
    default fn bar() -> u32 { 123 }
}

// END RUST SOURCE
// START rustc.main.Inline.before.mir
// let mut _0: ();
// let _1: u32;
// scope 1 {
//   debug x => _1;
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const <std::vec::Vec<()> as Foo>::bar() -> bb1;
// }
// bb1: {
//   _0 = ();
//   StorageDead(_1);
//   return;
// }
// END rustc.main.Inline.before.mir
// START rustc.main.Inline.after.mir
// let mut _0: ();
// let _1: u32;
// scope 1 {
//   debug x => _1;
// }
// scope 2 {
// }
// bb0: {
//   StorageLive(_1);
//   _1 = const 123u32;
//   _0 = ();
//   StorageDead(_1);
//   return;
// }
// END rustc.main.Inline.after.mir
