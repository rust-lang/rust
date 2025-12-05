//@ run-pass
//@ aux-build:cci_no_inline_lib.rs

extern crate cci_no_inline_lib;
use cci_no_inline_lib::iter;

pub fn main() {
    // Check that a cross-crate call function not marked as inline
    // does not, in fact, get inlined.  Also, perhaps more
    // importantly, checks that our scheme of using
    // sys::frame_address() to determine if we are inlining is
    // actually working.
    //let bt0 = sys::frame_address();
    //println!("%?", bt0);
    iter(vec![1, 2, 3], |i| {
        println!("{}", i);

        //let bt1 = sys::frame_address();
        //println!("%?", bt1);

        //assert!(bt0 != bt1);
    })
}
