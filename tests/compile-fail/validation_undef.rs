#![allow(unused_variables)]
// error-pattern: attempted to read undefined bytes

mod safe {
    use std::mem;

    pub(crate) fn make_float() -> f32 {
        unsafe { mem::uninitialized() }
    }
}

fn main() {
    let _x = safe::make_float();
}
