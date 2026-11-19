//! A generic field that the impl makes `&mut` counts as reborrowed, so a type with
//! two lifetimes is accepted, and borrowck still reborrows the field at the use site.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct Generic<'a, 'b, T> {
    t: T,
    shared: &'b (),
    marker: PhantomData<&'a ()>,
}

impl<'a, 'b> Reborrow for Generic<'a, 'b, &'a mut i32> {}

fn get<'a, 'b>(x: Generic<'a, 'b, &'a mut i32>) -> &'a mut i32 {
    x.t
}

fn main() {
    let mut v = 1;
    let u = ();
    let g = Generic { t: &mut v, shared: &u, marker: PhantomData };
    let y = get(g);
    let z = get(g); //~ ERROR cannot borrow `g` as mutable more than once at a time
    *y = 2;
    *z = 3;
}
