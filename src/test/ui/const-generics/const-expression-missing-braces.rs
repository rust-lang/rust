#![allow(incomplete_features)]
#![feature(const_generics)]

fn foo<const C: usize>() {}

const BAR: usize = 42;

fn a() {
    foo::<BAR + 3>(); //~ ERROR expected one of
}
fn b() {
    foo::<BAR + BAR>(); //~ ERROR likely `const` expression parsed as trait bounds
}
fn c() {
    foo::<3 + 3>(); //~ ERROR `const` generic expressions without braces are not supported
}
fn d() {
    foo::<BAR - 3>(); //~ ERROR expected one of
}
fn e() {
    foo::<BAR - BAR>(); //~ ERROR expected one of
}
fn f() {
    foo::<100 - BAR>(); //~ ERROR `const` generic expressions without braces are not supported
}
fn g() {
    foo::<bar<i32>()>(); //~ ERROR expected one of
}
fn h() {
    foo::<bar::<i32>()>(); //~ ERROR expected one of
}
fn i() {
    foo::<bar::<i32>() + BAR>(); //~ ERROR expected one of
}
fn j() {
    foo::<bar::<i32>() - BAR>(); //~ ERROR expected one of
}
fn k() {
    foo::<BAR - bar::<i32>()>(); //~ ERROR expected one of
}
fn l() {
    foo::<BAR - bar::<i32>()>(); //~ ERROR expected one of
}

const fn bar<const C: usize>() -> usize {
    C
}

fn main() {}
