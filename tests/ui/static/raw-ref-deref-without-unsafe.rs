use std::ptr;

// This code should remain unsafe because of the two unsafe operations here,
// even if in a hypothetical future we deem all &raw (const|mut) *ptr exprs safe.

static mut BYTE: u8 = 0;
static mut BYTE_PTR: *mut u8 = ptr::addr_of_mut!(BYTE);
// An unsafe static's ident is a place expression in its own right, so despite the above being safe
// (it's fine to create raw refs to places!) the following derefs the ptr before creating its ref!
static mut DEREF_BYTE_PTR: *mut u8 = ptr::addr_of_mut!(*BYTE_PTR);
//~^ ERROR: use of mutable static
//~| ERROR: dereference of raw pointer

fn main() {
    let _ = unsafe { DEREF_BYTE_PTR };
}
