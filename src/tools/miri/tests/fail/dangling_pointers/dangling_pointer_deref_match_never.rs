// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

#![allow(unreachable_code)]
#![feature(never_type)]

fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32 as *const !
    };
    unsafe {
        match *p {} //~ ERROR: entering unreachable code
    }
    panic!("this should never print");
}
