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
// START rustc.main.SimplifyCfg-elaborate-drops.after.mir
//     bb0: {
//         ...
//         _4 = &mut _2;
//         _3 = &raw mut (*_4);
//         ...
//         _6 = _3;
//         _5 = const foo(move _6) -> bb1;
//     }
//
//     bb1: {
//         ...
//         _7 = _2;
//         _8 = Len(_1);
//         _9 = Lt(_7, _8);
//         assert(move _9, "index out of bounds: the len is move _8 but the index is _7") -> bb2;
//     }
//
//     bb2: {
//         _1[_7] = move _5;
//         ...
//         return;
//     }
// END rustc.main.SimplifyCfg-elaborate-drops.after.mir
