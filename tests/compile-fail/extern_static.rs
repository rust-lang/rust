#![feature(raw_ref_op)]
//! Even referencing an unknown `extern static` already triggers an error.

extern "C" {
    static mut FOO: i32;
}

fn main() {
    let _val = unsafe { &raw const FOO }; //~ ERROR is not supported by Miri
}
