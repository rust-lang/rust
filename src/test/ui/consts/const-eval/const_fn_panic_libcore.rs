// check-pass

#![no_std]
#![crate_type = "lib"]
#![feature(const_panic)]

pub const fn always_panic() {
    panic!("always")
}

pub const fn assert_truth() {
    assert_eq!(2 + 2, 4)
}
