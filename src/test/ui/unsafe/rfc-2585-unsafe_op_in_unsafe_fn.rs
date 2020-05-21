#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
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

unsafe fn deny_level() {
    unsf();
    //~^ ERROR call to unsafe function is unsafe and requires unsafe block
    *PTR;
    //~^ ERROR dereference of raw pointer is unsafe and requires unsafe block
    VOID = ();
    //~^ ERROR use of mutable static is unsafe and requires unsafe block
    &PACKED.data;
    //~^ ERROR borrow of packed field is unsafe and requires unsafe block
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

// Check that `unsafe_op_in_unsafe_fn` works starting from the `warn` level.
#[warn(unsafe_op_in_unsafe_fn)]
#[deny(warnings)]
unsafe fn warning_level() {
    unsf();
    //~^ ERROR call to unsafe function is unsafe and requires unsafe block
    *PTR;
    //~^ ERROR dereference of raw pointer is unsafe and requires unsafe block
    VOID = ();
    //~^ ERROR use of mutable static is unsafe and requires unsafe block
    &PACKED.data;
    //~^ ERROR borrow of packed field is unsafe and requires unsafe block
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

unsafe fn explicit_block() {
    // no error
    unsafe {
        unsf();
        *PTR;
        VOID = ();
        &PACKED.data;
    }
}

unsafe fn two_explicit_blocks() {
    unsafe { unsafe { unsf() } }
    //~^ ERROR unnecessary `unsafe` block
}

#[warn(safe_packed_borrows)]
unsafe fn warn_packed_borrows() {
    &PACKED.data;
    //~^ WARNING borrow of packed field is unsafe and requires unsafe block
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

#[allow(safe_packed_borrows)]
unsafe fn allow_packed_borrows() {
    &PACKED.data; // `safe_packed_borrows` is allowed, no error
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn allow_level() {
    // lint allowed -> no error
    unsf();
    *PTR;
    VOID = ();
    &PACKED.data;

    unsafe { unsf() }
    //~^ ERROR unnecessary `unsafe` block
}

unsafe fn nested_allow_level() {
    #[allow(unsafe_op_in_unsafe_fn)]
    {
        // lint allowed -> no error
        unsf();
        *PTR;
        VOID = ();
        &PACKED.data;

        unsafe { unsf() }
        //~^ ERROR unnecessary `unsafe` block
    }
}

fn main() {
    unsf();
    //~^ ERROR call to unsafe function is unsafe and requires unsafe block
    #[allow(unsafe_op_in_unsafe_fn)]
    {
        unsf();
        //~^ ERROR call to unsafe function is unsafe and requires unsafe function or block
    }
}
