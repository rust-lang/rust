#![allow(dead_code)]
#![warn(clippy::transmute_null_to_fn)]
#![allow(clippy::zero_ptr)]

// Easy to lint because these only span one line.
fn one_liners() {
    unsafe {
        let _: fn() = std::mem::transmute(0 as *const ());
        //~^ ERROR: transmuting a known null pointer into a function pointer
        let _: fn() = std::mem::transmute(std::ptr::null::<()>());
        //~^ ERROR: transmuting a known null pointer into a function pointer
    }
}

pub const ZPTR: *const usize = 0 as *const _;
pub const NOT_ZPTR: *const usize = 1 as *const _;

fn transmute_const() {
    unsafe {
        // Should raise a lint.
        let _: fn() = std::mem::transmute(ZPTR);
        //~^ ERROR: transmuting a known null pointer into a function pointer
        // Should NOT raise a lint.
        let _: fn() = std::mem::transmute(NOT_ZPTR);
    }
}

fn issue_11485() {
    unsafe {
        let _: fn() = std::mem::transmute(0 as *const u8 as *const ());
        //~^ ERROR: transmuting a known null pointer into a function pointer
        let _: fn() = std::mem::transmute(std::ptr::null::<()>() as *const u8);
        //~^ ERROR: transmuting a known null pointer into a function pointer
        let _: fn() = std::mem::transmute(ZPTR as *const u8);
        //~^ ERROR: transmuting a known null pointer into a function pointer
    }
}

fn main() {
    one_liners();
    transmute_const();
}
