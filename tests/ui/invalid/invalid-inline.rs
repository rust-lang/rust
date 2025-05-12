#![allow(dead_code)]

#[inline(please,no)] //~ ERROR expected one argument
fn a() {
}

#[inline()] //~ ERROR expected one argument
fn b() {
}

fn main() {
    a();
    b();
}
