#![feature(nll)]
#![allow(warnings)]

fn foo<T: 'static>() { }

fn boo<'a>() {
    return;

    let x = foo::<&'a u32>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
