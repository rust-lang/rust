// revisions: no_flag with_flag
// [no_flag] check-pass
// [with_flag] compile-flags: -Zextra-const-ub-checks

use std::mem::transmute;

const INVALID_BOOL: () = unsafe {
    let _x: bool = transmute(3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| invalid value
};

const INVALID_PTR_IN_INT: () = unsafe {
    let _x: usize = transmute(&3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
};

const INVALID_SLICE_TO_USIZE_TRANSMUTE: () = unsafe {
    let x: &[u8] = &[0; 32];
    let _x: (usize, usize) = transmute(x);
    //[with_flag]~^ ERROR: evaluation of constant value failed
};

const UNALIGNED_PTR: () = unsafe {
    let _x: &u32 = transmute(&[0u8; 4]);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| invalid value
};

fn main() {}
