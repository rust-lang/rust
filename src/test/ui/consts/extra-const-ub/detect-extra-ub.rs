// revisions: no_flag with_flag
// [no_flag] check-pass
// [with_flag] compile-flags: -Zextra-const-ub-checks
#![feature(const_ptr_read)]

use std::mem::transmute;

const INVALID_BOOL: () = unsafe {
    let _x: bool = transmute(3u8);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| invalid value
};

const INVALID_PTR_IN_INT: () = unsafe {
    let _x: usize = transmute(&3u8);
    //[with_flag]~^ ERROR: any use of this value will cause an error
    //[with_flag]~| previously accepted
};

const INVALID_SLICE_TO_USIZE_TRANSMUTE: () = unsafe {
    let x: &[u8] = &[0; 32];
    let _x: (usize, usize) = transmute(x);
    //[with_flag]~^ ERROR: any use of this value will cause an error
    //[with_flag]~| previously accepted
};

const UNALIGNED_PTR: () = unsafe {
    let _x: &u32 = transmute(&[0u8; 4]);
    //[with_flag]~^ ERROR: evaluation of constant value failed
    //[with_flag]~| invalid value
};

const UNALIGNED_READ: () = {
    INNER; //[with_flag]~ERROR any use of this value will cause an error
    //[with_flag]~| previously accepted
    // There is an error here but its span is in the standard library so we cannot match it...
    // so we have this in a *nested* const, such that the *outer* const fails to use it.
    const INNER: () = unsafe {
        let x = &[0u8; 4];
        let ptr = x.as_ptr().cast::<u32>();
        ptr.read();
    };
};

fn main() {}
