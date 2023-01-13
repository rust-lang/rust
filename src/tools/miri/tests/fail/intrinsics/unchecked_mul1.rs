#![feature(unchecked_math)]
fn main() {
    // MAX overflow
    let _val = unsafe { 300u16.unchecked_mul(250u16) }; //~ ERROR: overflow executing `unchecked_mul`
}
