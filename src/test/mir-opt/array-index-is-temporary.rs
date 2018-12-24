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
//         _3 = &mut raw _2;
//         ...
//         _5 = _3;
//         _4 = const foo(move _5) -> bb1;
//     }
//
//     bb1: {
//         ...
//         _6 = _2;
//         _7 = Len(_1);
//         _8 = Lt(_6, _7);
//         assert(move _8, "index out of bounds: the len is move _7 but the index is _6") -> bb2;
//     }
//
//     bb2: {
//         _1[_6] = move _4;
//         ...
//         return;
//     }
// END rustc.main.EraseRegions.after.mir
