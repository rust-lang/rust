//@ check-pass

#![deny(unsafe_code)]

unsafe extern "C" {
    fn foo();
}

fn main() {}
