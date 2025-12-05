fn main() {
    // MAX overflow
    let _val = unsafe { 30000i16.unchecked_sub(-7000) }; //~ ERROR: arithmetic overflow in `unchecked_sub`
}
