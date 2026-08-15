fn main() {
    // MAX overflow
    let _val = unsafe { 40000u16.unchecked_add(30000) }; //~ ERROR: arithmetic overflow in `unchecked_add`
}
