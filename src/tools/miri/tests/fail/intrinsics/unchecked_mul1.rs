fn main() {
    // MAX overflow
    let _val = unsafe { 300u16.unchecked_mul(250u16) }; //~ ERROR: arithmetic overflow in `unchecked_mul`
}
