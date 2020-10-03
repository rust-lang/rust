#![feature(min_const_generics)]

fn foo<const C: usize>() {}

const BAR: usize = 42;

fn a() {
    foo::<BAR + 3>(); //~ ERROR expected one of
}
fn b() {
    // FIXME(const_generics): these diagnostics are awful, because trait objects without `dyn` were
    // a terrible mistake.
    foo::<BAR + BAR>();
    //~^ ERROR expected trait, found constant `BAR`
    //~| ERROR expected trait, found constant `BAR`
    //~| ERROR wrong number of const arguments: expected 1, found 0
    //~| ERROR wrong number of type arguments: expected 0, found 1
    //~| WARN trait objects without an explicit `dyn` are deprecated
}
fn c() {
    foo::<3 + 3>(); //~ ERROR expressions must be enclosed in braces
}
fn d() {
    foo::<BAR - 3>(); //~ ERROR expected one of
}
fn e() {
    foo::<BAR - BAR>(); //~ ERROR expected one of
}
fn f() {
    foo::<100 - BAR>(); //~ ERROR expressions must be enclosed in braces
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
