#![allow(dead_code)]

#[inline(please,no)] //~ ERROR malformed `inline` attribute
fn a() {
}

#[inline()] //~ ERROR malformed `inline` attribute
fn b() {
}

fn main() {
    a();
    b();
}
