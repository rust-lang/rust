#![feature(unchecked_math)]

fn main() {
    // MIN overflow
    let _val = unsafe { (-30000i16).unchecked_add(-8000) }; //~ ERROR: overflow executing `unchecked_add`
}
