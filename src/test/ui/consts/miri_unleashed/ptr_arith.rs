// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(core_intrinsics)]
#![allow(const_err)]

// A test demonstrating that we prevent doing even trivial
// pointer arithmetic or comparison during CTFE.

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
    //~| NOTE pointer-to-integer cast
};

static PTR_ARITH: () = unsafe {
    let x = &0 as *const _;
    let _v = core::intrinsics::offset(x, 0);
    //~^ ERROR could not evaluate static initializer
    //~| NOTE calling intrinsic `offset`
};

fn main() {}
