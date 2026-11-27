//@ check-pass
//@ run-rustfix

#![feature(const_volatile)]
#![allow(dead_code, unused_macros)]
#![warn(raw_borrows_via_references)]

const READ: () = unsafe {
    let x = 42i32;
    let y = (&x as *const i32).read_volatile();
    //~^ WARN creating an intermediate reference implies aliasing requirements
    assert!(x == y);
};

const WRITE: () = unsafe {
    let mut x = 42i32;
    (&mut x as *mut i32).write_volatile(13);
    //~^ WARN creating an intermediate reference implies aliasing requirements
    assert!(x == 13);
};

macro_rules! make_borrow {
    ($value:expr) => {
        &$value
    };
}

fn macro_generated_borrow() {
    let value = 0;
    // the scenario of macro should also get valid fix suggestion
    let _ = make_borrow!(value) as *const i32;
    //~^ WARN creating an intermediate reference implies aliasing requirements
}

fn main() {}
