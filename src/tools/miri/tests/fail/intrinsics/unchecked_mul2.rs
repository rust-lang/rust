#![feature(unchecked_math)]
fn main() {
    // MIN overflow
    let _val = unsafe { 1_000_000_000i32.unchecked_mul(-4) }; //~ ERROR: overflow executing `unchecked_mul`
}
