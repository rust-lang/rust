// Check that `MaybeDangling` actually prevents UB when it wraps dangling
// boxes and references
//
//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(maybe_dangling)]

use std::mem::{self, MaybeDangling};
use std::ptr::drop_in_place;

fn main() {
    boxy();
    reference();
    write_through_shared_ref();
}

fn boxy() {
    let mut x = MaybeDangling::new(Box::new(1));

    // make the box dangle
    unsafe { drop_in_place(x.as_mut()) };

    // move the dangling box (without `MaybeDangling` this causes UB)
    let x: MaybeDangling<Box<u32>> = x;

    mem::forget(x);
}

fn reference() {
    let x = {
        let local = 0;

        // erase the lifetime to make a dangling reference
        unsafe {
            mem::transmute::<MaybeDangling<&u32>, MaybeDangling<&u32>>(MaybeDangling::new(&local))
        }
    };

    // move the dangling reference (without `MaybeDangling` this causes UB)
    let _x: MaybeDangling<&u32> = x;
}

fn write_through_shared_ref() {
    // Under the current models, we do not forbid writing through
    // `MaybeDangling<&i32>`. That's not yet finally decided, but meanwhile
    // ensure we document this and notice when it changes.

    unsafe {
        let mutref = &mut 0;
        write_through_shr(mem::transmute(mutref));
    }

    fn write_through_shr(x: MaybeDangling<&i32>) {
        unsafe {
            let y: *mut i32 = mem::transmute(x);
            y.write(1);
        }
    }
}
