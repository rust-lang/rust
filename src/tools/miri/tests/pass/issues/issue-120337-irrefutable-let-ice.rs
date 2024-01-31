// Validation stops the test before the ICE we used to hit
//@compile-flags: -Zmiri-disable-validation

#![feature(never_type)]
#[derive(Copy, Clone)]
pub enum E {
    A(!),
}
pub union U {
    u: (),
    e: E,
}

fn main() {
    let E::A(ref _a) = unsafe { &(&U { u: () }).e };
}
