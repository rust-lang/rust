#![allow(dead_code)]
#![warn(clippy::transmuting_null)]
#![allow(clippy::zero_ptr)]
#![allow(clippy::transmute_ptr_to_ref)]
#![allow(clippy::eq_op, clippy::missing_transmute_annotations)]

// Easy to lint because these only span one line.
fn one_liners() {
    unsafe {
        let _: &u64 = std::mem::transmute(0 as *const u64);
        //~^ ERROR: transmuting a known null pointer into a reference
        //~| NOTE: `-D clippy::transmuting-null` implied by `-D warnings`
        let _: &u64 = std::mem::transmute(std::ptr::null::<u64>());
        //~^ ERROR: transmuting a known null pointer into a reference
    }
}

pub const ZPTR: *const usize = 0 as *const _;
pub const NOT_ZPTR: *const usize = 1 as *const _;

fn transmute_const() {
    unsafe {
        // Should raise a lint.
        let _: &u64 = std::mem::transmute(ZPTR);
        //~^ ERROR: transmuting a known null pointer into a reference
        // Should NOT raise a lint.
        let _: &u64 = std::mem::transmute(NOT_ZPTR);
    }
}

fn main() {
    one_liners();
    transmute_const();
}
