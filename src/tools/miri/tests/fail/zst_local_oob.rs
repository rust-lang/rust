fn main() {
    // make sure ZST locals cannot be accessed
    let x = &() as *const () as *const i8;
    let _val = unsafe { *x }; //~ ERROR: out-of-bounds
}
