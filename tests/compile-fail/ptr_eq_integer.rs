use std::mem;

fn main() {
    let b = Box::new(0);
    let x = &*b as *const i32;
    // We cannot compare this with a non-NULL integer. After all, these *could* be equal (with the right base address).
    assert!(x != mem::align_of::<i32>() as *const i32); //~ ERROR invalid arithmetic on pointers
}
