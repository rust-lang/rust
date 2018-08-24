#![feature(nll)]
#![allow(warnings)]

fn foo<'a>(x: &'a (u32,)) -> &'a u32 {
    let v = 22;
    &v
    //~^ ERROR `v` does not live long enough [E0597]
}

fn main() {}
