#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_unsafe)]

unsafe fn unsf() {}
const PTR: *const () = std::ptr::null();
static mut VOID: () = ();

unsafe fn deny_level() {
    unsf();
    //~^ ERROR call to unsafe function is unsafe and requires unsafe block
    *PTR;
    //~^ ERROR dereference of raw pointer is unsafe and requires unsafe block
    VOID = ();
    //~^ ERROR use of mutable static is unsafe and requires unsafe block

    unsafe {}
    //~^ ERROR unnecessary `unsafe` block
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
    unsafe {}
    //~^ ERROR unnecessary `unsafe` block
}

unsafe fn explicit_block() {
    // no error
    unsafe {
        unsf();
        *PTR;
        VOID = ();
    }
}

unsafe fn two_explicit_blocks() {
    unsafe { unsafe { unsf() } }
    //~^ ERROR unnecessary `unsafe` block
}

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn allow_level() {
    // lint allowed -> no error
    unsf();
    *PTR;
    VOID = ();

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
