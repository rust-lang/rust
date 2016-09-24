fn main() {
    let x = &() as *const () as *const i32;
    let _ = unsafe { *x }; //~ ERROR: tried to access memory through an invalid pointer
}
