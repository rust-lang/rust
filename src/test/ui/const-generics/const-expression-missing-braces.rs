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
    //~^ ERROR likely `const` expression parsed as trait bounds
}
fn c() {
    let bar = 3;
    foo::<3 + 3>();
    //~^ ERROR expected one of `,` or `>`, found `+`
}

fn main() {}
