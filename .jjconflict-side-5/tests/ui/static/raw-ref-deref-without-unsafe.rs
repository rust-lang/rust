use std::ptr;

static mut BYTE: u8 = 0;
static mut BYTE_PTR: *mut u8 = ptr::addr_of_mut!(BYTE);

// This code should remain unsafe because reading from a static mut is *always* unsafe.

// An unsafe static's ident is a place expression in its own right, so despite the above being safe
// (it's fine to create raw refs to places!) the following derefs the ptr before creating its ref!
static mut DEREF_BYTE_PTR: *mut u8 = ptr::addr_of_mut!(*BYTE_PTR);
//~^ ERROR: use of mutable static

fn main() {
    let _ = unsafe { DEREF_BYTE_PTR };
}
