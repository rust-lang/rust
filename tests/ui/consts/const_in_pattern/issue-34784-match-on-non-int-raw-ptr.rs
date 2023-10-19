#![deny(pointer_structural_match)]
#![allow(dead_code)]
const C: *const u8 = &0;

fn foo(x: *const u8) {
    match x {
        C => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }
}

const D: *const [u8; 4] = b"abcd";

fn main() {
    match D {
        D => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }
}
