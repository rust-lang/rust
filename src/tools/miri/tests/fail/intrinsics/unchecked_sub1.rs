fn main() {
    // MIN overflow
    let _val = unsafe { 14u32.unchecked_sub(22) }; //~ ERROR: arithmetic overflow in `unchecked_sub`
}
