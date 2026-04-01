#![feature(fn_delegation)]
#![allow(incomplete_features)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_unsafe)]

mod to_reuse {
    unsafe extern "C" {
        pub fn default_unsafe_foo();
        pub unsafe fn unsafe_foo();
        pub safe fn safe_foo();
    }
}

reuse to_reuse::{default_unsafe_foo, unsafe_foo, safe_foo};

fn main() {
    let _: extern "C" fn() = default_unsafe_foo;
    //~^ ERROR mismatched types
    let _: extern "C" fn() = unsafe_foo;
    //~^ ERROR mismatched types
    let _: extern "C" fn() = safe_foo;
}
