#![feature(nll)]
#![allow(warnings)]

fn gimme(x: &(u32,)) -> &u32 {
    &x.0
}

fn foo<'a>(x: &'a (u32,)) -> &'a u32 {
    let v = 22;
    gimme(&(v,))
    //~^ ERROR borrowed value does not live long enough [E0597]
}

fn main() {}
