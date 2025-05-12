#![feature(never_type)]

fn make_up_a_value<T>() -> T {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        &raw const *x;
        // Since `*x` is `!`, HIR typeck used to think that it diverges
        // and allowed the block to coerce to any value, leading to UB.
    }
}


fn make_up_a_pointer<T>() -> *const T {
    unsafe {
        let x: *const ! = 0 as _;
        &raw const *x
        //~^ ERROR mismatched types
    }
}

fn main() {}
