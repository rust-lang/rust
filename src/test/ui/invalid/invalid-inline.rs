#![allow(dead_code)]

#[inline(please_no)] //~ ERROR invalid argument
fn a() {
}

#[inline(please,no)] //~ ERROR expected one argument
fn b() {
}

#[inline()] //~ ERROR expected one argument
fn c() {
}

fn main() {
    a();
    b();
    c();
}
