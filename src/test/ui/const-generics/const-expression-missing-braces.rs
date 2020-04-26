#![allow(incomplete_features)]
#![feature(const_generics)]

fn foo<const C: usize>() {}

const BAR: usize = 42;

fn a() {
    foo::<BAR + 3>();
    //~^ ERROR expected one of
}
fn b() {
    foo::<BAR + BAR>();
    //~^ ERROR likely `const` expression parsed as trait bounds
}
fn c() {
    foo::<3 + 3>(); // ok
}
fn d() {
    foo::<BAR - 3>();
    //~^ ERROR expected one of
}
fn e() {
    foo::<BAR - BAR>();
    //~^ ERROR expected one of
}
fn f() {
    foo::<100 - BAR>(); // ok
}
fn main() {}
