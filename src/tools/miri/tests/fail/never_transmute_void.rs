// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation

#![feature(never_type)]
#![allow(unused, invalid_value)]

mod m {
    enum VoidI {}
    pub struct Void(VoidI);

    pub fn f(v: Void) -> ! {
        match v.0 {}
        //~^ ERROR: entering unreachable code
    }
}

fn main() {
    let v = unsafe { std::mem::transmute::<(), m::Void>(()) };
    m::f(v);
}
