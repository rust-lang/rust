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

struct Foo(i32, String, u32);

// EMIT_MIR write_via_move.my_write_unaligned.CleanupPostBorrowck.after.mir
// CHECK-LABEL: fn my_write_unaligned
#[inline(never)]
unsafe fn my_write_unaligned(dst: *mut Foo, src: String) {
    // CHECK: [[TEMP:_.+]] = copy _1;
    // CHECK: ((*[[TEMP]]).1: std::string::String) = move _2;
    unsafe { std::intrinsics::write_field_via_move::<_, _, 1>(dst, src) }
}

fn main() {
    box_new(0);

    let mut foo = Foo(0, String::new(), 0);
    unsafe { my_write_unaligned(&raw mut foo, String::new()) };
}
