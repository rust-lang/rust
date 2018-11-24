use std::cell::Cell;

const FOO: &u32 = {
    let mut a = 42;
    {
        let b: *mut u32 = &mut a; //~ ERROR may only refer to immutable values
        unsafe { *b = 5; } //~ ERROR dereferencing raw pointers in constants
        //~^ ERROR constant contains unimplemented expression
    }
    &{a}
};

fn main() {}
