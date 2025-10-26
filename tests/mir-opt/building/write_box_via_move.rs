//! Ensure we don't generate unnecessary copys for `write_via_move`.
//@ compile-flags: -Zmir-opt-level=0
#![feature(liballoc_internals)]

extern crate alloc;

// Can't emit `built.after` here as that contains user type annotations which contain DefId that
// change all the time.
// EMIT_MIR write_box_via_move.box_new.CleanupPostBorrowck.after.mir
// CHECK-LABEL: fn box_new
#[inline(never)]
fn box_new<T: Copy>(x: T) -> Box<[T; 1024]> {
    let mut b = Box::new_uninit();
    // Ensure the array gets constructed directly into the deref'd pointer.
    // CHECK: (*[[TEMP1:_.+]]) = [{{(move|copy) _.+}}; 1024];
    unsafe { alloc::intrinsics::write_box_via_move(b, [x; 1024]).assume_init() }
}

// EMIT_MIR write_box_via_move.vec_macro.CleanupPostBorrowck.after.mir
// CHECK-LABEL: fn vec_macro
fn vec_macro() -> Vec<i32> {
    // CHECK: (*[[TEMP1:_.+]]) = [const 0_i32, const 1_i32,
    vec![0, 1, 2, 3, 4, 5, 6, 7]
}

fn main() {
    box_new(0);
    vec_macro();
}
