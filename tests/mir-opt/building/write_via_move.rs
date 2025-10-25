//! Ensure we don't generate unnecessary copys for `write_via_move`.
//@ compile-flags: -Zmir-opt-level=0
#![feature(core_intrinsics)]

use std::mem;

// Can't emit `built.after` here as that contains user type annotations which contain DefId that
// change all the time.
// EMIT_MIR write_via_move.box_new.CleanupPostBorrowck.after.mir
// CHECK-LABEL: fn box_new
#[inline(never)]
fn box_new<T: Copy>(x: T) -> Box<[T; 1024]> {
    let mut b = Box::new_uninit();
    let ptr = mem::MaybeUninit::as_mut_ptr(&mut *b);
    // Ensure the array gets constructed directly into the deref'd pointer.
    // CHECK: (*[[TEMP1:_.+]]) = [{{(move|copy) _.+}}; 1024];
    unsafe { std::intrinsics::write_via_move(ptr, [x; 1024]) };
    unsafe { b.assume_init() }
}

fn main() {
    box_new(0);
}
