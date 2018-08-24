// Check that array elemen types must be Sized. Issue #25692.

#![feature(rustc_attrs)]
#![allow(dead_code)]

struct Foo {
    foo: [[u8]], //~ ERROR E0277
}

#[rustc_error]
fn main() { }
