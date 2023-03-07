#![allow(invalid_value)]

fn main() {
    // The array avoids a `Scalar` layout which detects uninit without even doing validation.
    let _val = unsafe { std::mem::MaybeUninit::<[*const u8; 1]>::uninit().assume_init() };
    //~^ ERROR: uninitialized
}
