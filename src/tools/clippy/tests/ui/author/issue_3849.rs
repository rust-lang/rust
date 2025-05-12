//@ check-pass

#![allow(dead_code)]
#![allow(clippy::zero_ptr)]
#![allow(clippy::transmute_ptr_to_ref)]
#![allow(clippy::transmuting_null, clippy::missing_transmute_annotations)]

pub const ZPTR: *const usize = 0 as *const _;

fn main() {
    unsafe {
        #[clippy::author]
        let _: &i32 = std::mem::transmute(ZPTR);
        let _: &i32 = std::mem::transmute(0 as *const i32);
    }
}
