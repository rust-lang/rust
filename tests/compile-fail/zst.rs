fn main() {
    let x = &() as *const () as *const i32;
    let _ = unsafe { *x }; //~ ERROR outside bounds of allocation
}
