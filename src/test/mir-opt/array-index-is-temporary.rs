// Retagging (from Stacked Borrows) relies on the array index being a fresh
// temporary, so that side-effects cannot change it.
// Test that this is indeed the case.

unsafe fn foo(z: *mut usize) -> u32 {
    *z = 2;
    99
}

fn main() {
    let mut x = [42, 43, 44];
    let mut y = 1;
    let z: *mut usize = &mut y;
    x[y] = unsafe { foo(z) };
}

// END RUST SOURCE
// START rustc.main.EraseRegions.after.mir
//     bb0: {
//         ...
//         _5 = &mut _2;
//         _4 = &mut (*_5);
//         _3 = move _4 as *mut usize (Misc);
//         ...
//         _7 = _3;
//         _6 = const foo(move _7) -> bb1;
//     }
//
//     bb1: {
//         ...
//         _8 = _2;
//         _9 = Len(_1);
//         _10 = Lt(_8, _9);
//         assert(move _10, "index out of bounds: the len is move _9 but the index is _8") -> bb2;
//     }
//
//     bb2: {
//         _1[_8] = move _6;
//         ...
//         return;
//     }
// END rustc.main.EraseRegions.after.mir
