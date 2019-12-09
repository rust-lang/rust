// ignore-wasm32-bare compiled with panic=abort by default
// ignore-tidy-linelength
// compile-flags: -Z mir-emit-retag -Z mir-opt-level=0 -Z span_free_formats

#![allow(unused)]

struct Test(i32);

impl Test {
    // Make sure we run the pass on a method, not just on bare functions.
    fn foo<'x>(&self, x: &'x mut i32) -> &'x mut i32 { x }
    fn foo_shr<'x>(&self, x: &'x i32) -> &'x i32 { x }
}

impl Drop for Test {
    fn drop(&mut self) {}
}

fn main() {
    let mut x = 0;
    {
        let v = Test(0).foo(&mut x); // just making sure we do not panic when there is a tuple struct ctor
        let w = { v }; // assignment
        let w = w; // reborrow
        // escape-to-raw (mut)
        let _w = w as *mut _;
    }

    // Also test closures
    let c: fn(&i32) -> &i32 = |x: &i32| -> &i32 { let _y = x; x };
    let _w = c(&x);

    // need to call `foo_shr` or it doesn't even get generated
    Test(0).foo_shr(&0);

    // escape-to-raw (shr)
    let _w = _w as *const _;
}

// END RUST SOURCE
// START rustc.{{impl}}-foo.EraseRegions.after.mir
//     bb0: {
//         Retag([fn entry] _1);
//         Retag([fn entry] _2);
//         ...
//         _0 = &mut (*_3);
//         Retag(_0);
//         ...
//         return;
//     }
// END rustc.{{impl}}-foo.EraseRegions.after.mir
// START rustc.{{impl}}-foo_shr.EraseRegions.after.mir
//     bb0: {
//         Retag([fn entry] _1);
//         Retag([fn entry] _2);
//         ...
//         _0 = _2;
//         Retag(_0);
//         ...
//         return;
//     }
// END rustc.{{impl}}-foo_shr.EraseRegions.after.mir
// START rustc.main.EraseRegions.after.mir
// fn main() -> () {
//     ...
//     bb0: {
//         ...
//         _3 = const Test::foo(move _4, move _6) -> [return: bb2, unwind: bb3];
//     }
//
//     ...
//
//     bb2: {
//         Retag(_3);
//         ...
//         _9 = move _3;
//         Retag(_9);
//         _8 = &mut (*_9);
//         Retag(_8);
//         StorageDead(_9);
//         StorageLive(_10);
//         _10 = move _8;
//         Retag(_10);
//         ...
//         _13 = &mut (*_10);
//         Retag(_13);
//         _12 = move _13 as *mut i32 (Misc);
//         Retag([raw] _12);
//         ...
//         _16 = move _17(move _18) -> bb5;
//     }
//
//     bb5: {
//         Retag(_16);
//         ...
//         _20 = const Test::foo_shr(move _21, move _23) -> [return: bb6, unwind: bb7];
//     }
//
//     ...
// }
// END rustc.main.EraseRegions.after.mir
// START rustc.main-{{closure}}.EraseRegions.after.mir
// fn main::{{closure}}#0(_1: &[closure@main::{{closure}}#0], _2: &i32) -> &i32 {
//     ...
//     bb0: {
//         Retag([fn entry] _1);
//         Retag([fn entry] _2);
//         StorageLive(_3);
//         _3 = _2;
//         Retag(_3);
//         _0 = _2;
//         Retag(_0);
//         StorageDead(_3);
//         return;
//     }
// }
// END rustc.main-{{closure}}.EraseRegions.after.mir
// START rustc.ptr-real_drop_in_place.Test.SimplifyCfg-make_shim.after.mir
// fn  std::ptr::real_drop_in_place(_1: &mut Test) -> () {
//     ...
//     bb0: {
//         Retag([raw] _1);
//         _2 = &mut (*_1);
//         _3 = const <Test as std::ops::Drop>::drop(move _2) -> bb1;
//     }
//
//     bb1: {
//         return;
//     }
// }
// END rustc.ptr-real_drop_in_place.Test.SimplifyCfg-make_shim.after.mir
