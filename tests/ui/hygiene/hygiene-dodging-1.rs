// run-pass
#![allow(unused_must_use)]

mod x {
    pub fn g() -> usize {14}
}

pub fn main(){
    // should *not* shadow the module x:
    let x = 9;
    // use it to avoid warnings:
    x+3;
    assert_eq!(x::g(),14);
}
