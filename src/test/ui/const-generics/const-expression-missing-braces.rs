#![allow(incomplete_features)]
#![feature(const_generics)]

fn foo<const C: usize>() {}

fn a() {
    let bar = 3;
    foo::<bar + 3>();
    //~^ ERROR expected one of `!`, `(`, `,`, `>`, `?`, `for`, lifetime, or path, found `3`
}
fn b() {
    let bar = 3;
    foo::<bar + bar>();
    //~^ ERROR expected trait, found local variable `bar`
    //~| ERROR expected trait, found local variable `bar`
    //~| ERROR wrong number of const arguments: expected 1, found 0
    //~| ERROR wrong number of type arguments: expected 0, found 1
    //~| WARNING trait objects without an explicit `dyn` are deprecated
}
fn c() {
    let bar = 3;
    foo::<3 + 3>();
    //~^ ERROR expected one of `,` or `>`, found `+`
}

fn main() {}
