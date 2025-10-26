use std::num::NonZero;

#[repr(C)]
struct S1(NonZero<i32>);

#[repr(C)]
struct S2(i32);

fn callee(_s: S2) {}

fn main() {
    let fnptr: fn(S2) = callee;
    let fnptr: fn(S1) = unsafe { std::mem::transmute(fnptr) };
    fnptr(S1(NonZero::new(1).unwrap()));
    //~^ ERROR: type S2 passing argument of type S1
}
