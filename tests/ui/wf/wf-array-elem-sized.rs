// Check that array element types must be Sized. Issue #25692.


#![allow(dead_code)]

struct Foo {
    foo: [[u8]], //~ ERROR E0277
}


fn main() { }
