// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(core_intrinsics)]
#![allow(const_err)]

// During CTFE, we prevent pointer comparison and pointer-to-int casts.

static CMP: () = {
    let x = &0 as *const _;
    let _v = x == x;
    //~^ ERROR could not evaluate static initializer
    //~| NOTE pointer arithmetic or comparison
};

static INT_PTR_ARITH: () = unsafe {
    let x: usize = std::mem::transmute(&0);
    let _v = x + 0;
    //~^ ERROR could not evaluate static initializer
    //~| NOTE cannot cast pointer to integer
};

fn main() {}
