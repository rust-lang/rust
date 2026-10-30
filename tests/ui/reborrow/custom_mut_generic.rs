//! A generic parameter implementing Reborrow cannot currently
//! be reborrowed multiple times.

#![feature(reborrow)]
use std::marker::Reborrow;

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}

fn method(_: impl Reborrow) {}

fn generic(_: impl Sized) {}

fn generic_reborrow(a: impl Reborrow) {
    let _ = method(a);
    let _ = method(a); //~ ERROR use of moved value

    generic(a); //~ ERROR use of moved value
    generic(a); //~ ERROR use of moved value
    {
        a  //~ ERROR use of moved value
    };
    generic(a); //~ ERROR use of moved value
    let _local = a; //~ ERROR use of moved value
    generic(a); //~ ERROR use of moved value
    let _ = || a; //~ ERROR use of moved value
    generic(a); //~ ERROR use of moved value
}

fn main() {
    generic_reborrow(CustomMut(&mut ()));
}
