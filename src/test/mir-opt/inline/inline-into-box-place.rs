// ignore-tidy-linelength
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -Z mir-opt-level=3
// only-64bit FIXME: the mir representation of RawVec depends on ptr size
#![feature(box_syntax)]

fn main() {
    let _x: Box<Vec<u32>> = box Vec::new();
}

// END RUST SOURCE
// START rustc.main.Inline.before.mir
// let mut _0: ();
// let _1: std::boxed::Box<std::vec::Vec<u32>> as UserTypeProjection { base: UserType(0), projs: [] };
// let mut _2: std::boxed::Box<std::vec::Vec<u32>>;
// let mut _3: ();
// scope 1 {
//   debug _x => _1;
// }
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = Box(std::vec::Vec<u32>);
//   (*_2) = const std::vec::Vec::<u32>::new() -> [return: bb2, unwind: bb4];
// }
// bb1 (cleanup): {
//   resume;
// }
// bb2: {
//   _1 = move _2;
//   StorageDead(_2);
//   _0 = ();
//   drop(_1) -> [return: bb3, unwind: bb1];
// }
// bb3: {
//   StorageDead(_1);
//   return;
// }
// bb4 (cleanup): {
//   _3 = const alloc::alloc::box_free::<std::vec::Vec<u32>>(move (_2.0: std::ptr::Unique<std::vec::Vec<u32>>)) -> bb1;
// }
// END rustc.main.Inline.before.mir
// START rustc.main.Inline.after.mir
// let mut _0: ();
// let _1: std::boxed::Box<std::vec::Vec<u32>> as UserTypeProjection { base: UserType(0), projs: [] };
// let mut _2: std::boxed::Box<std::vec::Vec<u32>>;
// let mut _3: ();
// let mut _4: &mut std::vec::Vec<u32>;
// scope 1 {
//   debug _x => _1;
// }
// scope 2 {
// }
// bb0: {
//   StorageLive(_1);
//   StorageLive(_2);
//   _2 = Box(std::vec::Vec<u32>);
//   _4 = &mut (*_2);
//   ((*_4).0: alloc::raw_vec::RawVec<u32>) = const ByRef { alloc: Allocation { bytes: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], relocations: Relocations(SortedMap { data: [] }), undef_mask: UndefMask { blocks: [65535], len: Size { raw: 16 } }, size: Size { raw: 16 }, align: Align { pow2: 3 }, mutability: Not, extra: () }, offset: Size { raw: 0 } }: alloc::raw_vec::RawVec::<u32>;
//   ((*_4).1: usize) = const 0usize;
//   _1 = move _2;
//   StorageDead(_2);
//   _0 = ();
//   drop(_1) -> [return: bb2, unwind: bb1];
// }
// bb1 (cleanup): {
//   resume;
// }
// bb2: {
//   StorageDead(_1);
//   return;
// }
// END rustc.main.Inline.after.mir
