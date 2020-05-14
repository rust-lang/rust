#![feature(unsafe_block_in_unsafe_fn)]
#![warn(unsafe_op_in_unsafe_fn)]
#![deny(unused_unsafe)]
#![deny(safe_packed_borrows)]

unsafe fn unsf() {}
const PTR: *const () = std::ptr::null();
static mut VOID: () = ();

#[repr(packed)]
pub struct Packed {
    data: &'static u32,
}

const PACKED: Packed = Packed { data: &0 };

unsafe fn foo() {
    unsf();
    //~^ WARNING call to unsafe function is unsafe and requires unsafe block
    *PTR;
    //~^ WARNING dereference of raw pointer is unsafe and requires unsafe block
    VOID = ();
    //~^ WARNING use of mutable static is unsafe and requires unsafe block
    &PACKED.data; // the level for the `safe_packed_borrows` lint is ignored
    //~^ WARNING borrow of packed field is unsafe and requires unsafe block
}

unsafe fn bar() {
    // no error
    unsafe {
        unsf();
        *PTR;
        VOID = ();
        &PACKED.data;
    }
}

unsafe fn baz() {
    unsafe { unsafe { unsf() } }
    //~^ ERROR unnecessary `unsafe` block
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn qux() {
    // lint allowed -> no error
    unsf();
    *PTR;
    VOID = ();
    &PACKED.data;

    unsafe { unsf() }
    //~^ ERROR unnecessary `unsafe` block
}

fn main() {
    unsf()
    //~^ ERROR call to unsafe function is unsafe and requires unsafe function or block
}
