// This should fail even without validation
// compile-flags: -Zmiri-disable-validation

#![feature(never_type)]
#![allow(unused, invalid_value)]

enum Void {}

fn f(v: Void) -> ! {
    match v {} //~ ERROR entering unreachable code
}

fn main() {
    let v: Void = unsafe {
        std::mem::transmute::<(), Void>(())
    };
    f(v); //~ inside `main`
}
