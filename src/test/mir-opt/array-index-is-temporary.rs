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
//         _6 = &mut _2;
//         _5 = &mut (*_6);
//         _4 = move _5 as *mut usize (Misc);
//         _3 = move _4;
//         ...
//         _8 = _3;
//         _7 = const foo(move _8) -> bb1;
//     }
//
//     bb1: {
//         ...
//         _9 = _2;
//         _10 = Len(_1);
//         _11 = Lt(_9, _10);
//         assert(move _11, "index out of bounds: the len is move _10 but the index is _9") -> bb2;
//     }
//
//     bb2: {
//         _1[_9] = move _7;
//         ...
//         return;
//     }
// END rustc.main.EraseRegions.after.mir
