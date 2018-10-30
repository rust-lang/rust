use std::mem;

// Make sure we cannot use raw ptrs to access a local that
// has never been escaped to the raw world.
fn main() {
    let mut x = 42;
    let ptr = &mut x;
    let raw: *mut i32 = unsafe { mem::transmute(ptr) };
    unsafe { *raw = 13; } //~ ERROR does not exist on the stack
}
