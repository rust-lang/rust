#![deny(pointer_structural_match)]
#![allow(dead_code)]

const C: *const u8 = &0;
// Make sure we also find pointers nested in other types.
const C_INNER: (*const u8, u8) = (C, 0);

fn foo(x: *const u8) {
    match x {
        C => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }
}

fn foo2(x: *const u8) {
    match (x, 1) {
        C_INNER => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }
}

const D: *const [u8; 4] = b"abcd";

const STR: *const str = "abcd";

fn main() {
    match D {
        D => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }

    match STR {
        STR => {} //~ERROR: behave unpredictably
        //~| previously accepted
        _ => {}
    }
}
